"""v3 extension of the copied v2 world model. No second encoder/ODE for ADT."""
from dataclasses import dataclass,asdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from unified_model_v3 import Config,WorldModelV3,Encoder

def representation_matrix(esm,kind,seed=731):
    """Sequence controls match the reference's coverage as well as dimension/scale."""
    out=np.asarray(esm,np.float32).copy();covered=np.linalg.norm(out,axis=1)>0
    rng=np.random.default_rng(seed)
    if kind=='random':
        out[covered]=rng.normal(size=out[covered].shape).astype('f')
        out/=np.maximum(np.linalg.norm(out,axis=0,keepdims=True),1e-8)
    elif kind=='shuffled_esm':
        ids=np.flatnonzero(covered);out[ids]=out[rng.permutation(ids)]
    return out

@dataclass
class V3Config(Config):
    cond_dim: int = 12
    n_proteins: int = 228
    protein_weight: float = 1.0
    protein_encoder_grad: bool = True
    representation: str = "esm"
    mapping: str = "ode"
    stochastic: bool = True
    species_condition: bool = True
    disease_condition: bool = True
    source_expression: bool = True
    objective: str = "distribution"
    beta_recon: float = 0.1
    beta_kl: float = 0.01
    beta_drift: float = 0.01
    separate_rev_drift: bool = True
    beta_rev: float = 1.0

def mmd(x,y,scale):
    """Biased, nonnegative multi-bandwidth RBF MMD, one frozen cost scale."""
    def kernel(a,b):
        d=(a.square().sum(1)[:,None]+b.square().sum(1)[None,:]-2*a@b.T).clamp_min(0)/scale
        return sum(torch.exp(-d/w) for w in (0.5,1.,2.))/3
    return kernel(x,x).mean()+kernel(y,y).mean()-2*kernel(x,y).mean()

class WorldModel(WorldModelV3):
    def __init__(self,esm,mean,protein_mean,protein_sd,cfg):
        super().__init__(len(mean),np.asarray(esm,np.float32),cfg)
        self.gene_mean.copy_(torch.as_tensor(mean,dtype=torch.float32))
        self.protein_head=nn.Sequential(nn.Linear(cfg.latent_dim,cfg.enc_hidden),nn.ReLU(),
                                        nn.Linear(cfg.enc_hidden,cfg.n_proteins))
        self.register_buffer("protein_mean",torch.as_tensor(protein_mean,dtype=torch.float32))
        self.register_buffer("protein_sd",torch.as_tensor(protein_sd,dtype=torch.float32))
        self.raw_projection=nn.Linear(self.n_genes,self.esm_dim,bias=False) if cfg.representation=="learned" else None
        if cfg.representation=="learned":
            # Remove sequence information from decoder initialization too.
            self.dec_out.reset_parameters()
        self.mlp_mapping=nn.Sequential(nn.Linear(cfg.latent_dim+cfg.cond_dim,cfg.ode_hidden),
                                       nn.ReLU(),nn.Linear(cfg.ode_hidden,cfg.latent_dim))
        # brain->blood exists solely for source compatibility; not optimized by this trainer.
        for p in self.drift_net.parameters():p.requires_grad_(False)
        if cfg.mapping!="mlp":
            for p in self.mlp_mapping.parameters():p.requires_grad_(False)
        if cfg.mapping!="ode" and self.drift_net_rev is not None:
            for p in self.drift_net_rev.parameters():p.requires_grad_(False)
        if not cfg.protein_weight:
            for p in self.protein_head.parameters():p.requires_grad_(False)

    def condition(self,c):
        c=c.clone()
        if not self.config.species_condition:c[:,:2]=0
        if not self.config.disease_condition:c[:,2:4]=0
        return c

    def encode(self,x,sample=False,mask=None):
        xc=x-self.gene_mean
        if mask is not None:xc=xc*mask
        h=self.raw_projection(xc) if self.raw_projection is not None else xc@self.esm_matrix
        mu,lv=self.encoder(h)
        z=Encoder.reparam(mu,lv) if sample and self.config.stochastic else mu
        return z,mu,lv

    def decode(self,z):
        return self.dec_out(self.decoder_z(z))+self.gene_mean

    def predict_brain(self,x,c,sample=False,noise=None,source_mask=None):
        z,mu,lv=self.encode(x,sample,source_mask)
        if not self.config.source_expression:
            if noise is not None:z=noise
            elif self.training:z=torch.randn_like(z)
            else:
                gen=torch.Generator(device=z.device).manual_seed(731)
                z=torch.randn(z.shape,device=z.device,dtype=z.dtype,generator=gen)
            mu=z
        c=self.condition(c)
        if self.config.mapping=="mlp":zt=z+self.mlp_mapping(torch.cat([z,c],1))
        elif self.config.mapping=="identity":zt=z
        else:zt=self.forward_ode(z,"blood_to_brain",c)
        return self.decode(zt),dict(mu=mu,lv=lv,z=z,zt=zt)

    def predict_protein(self,x,detach_encoder=False,mask=None):
        z,_,_=self.encode(x,False,mask)
        if detach_encoder:z=z.detach()
        return self.protein_head(z)

    def predict_adt(self,x,mask=None):
        return self.predict_protein(x,mask=mask)*self.protein_sd+self.protein_mean

    def losses(self,xb,yb,c,xc,uc,cost_scale,shuffle_protein=False,source_mask=None,target_mask=None,cite_mask=None):
        cfg=self.config
        pred,d=self.predict_brain(xb,c,sample=True,source_mask=source_mask)
        idx=target_mask.bool() if target_mask is not None else slice(None)
        if cfg.objective=="legacy_random_pair":
            target=yb[:len(pred)]
            dist=F.mse_loss(pred[:,idx],target[:,idx])
        else:dist=mmd(pred[:,idx],yb[:,idx],cost_scale)+0.1*F.mse_loss(pred[:,idx].mean(0),yb[:,idx].mean(0))
        zb,mb,lb=self.encode(xb,True,source_mask)
        zy,my,ly=self.encode(yb,True,target_mask)
        def reconstruct_loss(pred,target,mask):
            weight=1+(cfg.zero_weight-1)*(target.abs()>1e-6).to(target.dtype)
            if mask is not None:
                return ((pred-target).square()*weight*mask).sum()/(len(target)*mask.sum().clamp_min(1))
            return ((pred-target).square()*weight).mean()
        recon=0.5*(reconstruct_loss(self.decode(zb),xb,source_mask)+reconstruct_loss(self.decode(zy),yb,target_mask))
        kl=0.5*(self.kl_loss(mb,lb)+self.kl_loss(my,ly)) if cfg.stochastic else pred.new_zeros(())
        drift=(d["zt"]-d["z"]).square().mean()
        prot=pred.new_zeros(())
        if cfg.protein_weight:
            u=(uc-self.protein_mean)/self.protein_sd
            if shuffle_protein:u=u[torch.randperm(len(u),device=u.device)]
            prot=F.mse_loss(self.predict_protein(xc,not cfg.protein_encoder_grad,mask=cite_mask),u)
        total=dist+cfg.beta_recon*recon+cfg.beta_kl*kl+cfg.beta_drift*drift+cfg.protein_weight*prot
        return dict(loss=total,distribution=dist,recon=recon,kl=kl,drift=drift,protein=prot)

    @torch.no_grad()
    def perturb(self,x,c,gene,value=0.,source_mask=None):
        if self.training:raise RuntimeError("Perturbation requires eval() for deterministic inference.")
        base,_=self.predict_brain(x,c,False,source_mask=source_mask)
        q=x.clone();q[:,gene]=value
        out,_=self.predict_brain(q,c,False,source_mask=source_mask)
        return out-base,self.predict_adt(q,source_mask)-self.predict_adt(x,source_mask)
