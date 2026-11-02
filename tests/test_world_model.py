import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/step1_world_model'))
import numpy as np,torch
from world_model import WorldModel,V3Config,mmd

torch.set_num_threads(2)
def model(**kw):
 torch.manual_seed(11)
 cfg=V3Config(latent_dim=8,enc_hidden=16,dec_z_hidden=16,ode_hidden=16,
              ode_n_steps=2,n_proteins=5,cond_dim=12,**kw)
 return WorldModel(np.random.default_rng(1).normal(size=(24,12)).astype('f'),
                   np.ones(24,'f'),np.zeros(5,'f'),np.ones(5,'f'),cfg)
def inputs():
 return torch.rand(8,24),torch.rand(8,24),torch.eye(12)[:8],torch.rand(8,24),torch.rand(8,5)
def test_protein_gradient_reaches_shared_encoder():
 m=model();x,y,c,xc,u=inputs()
 m.losses(x,y,c,xc,u,torch.tensor(24.))['protein'].backward()
 assert sum(p.grad.abs().sum() for p in m.encoder.parameters() if p.grad is not None)>0
 assert sum(p.grad.abs().sum() for p in m.protein_head.parameters() if p.grad is not None)>0
def test_detached_head_has_no_encoder_gradient():
 m=model(protein_encoder_grad=False);x,y,c,xc,u=inputs()
 m.losses(x,y,c,xc,u,torch.tensor(24.))['protein'].backward()
 assert all(p.grad is None for p in m.encoder.parameters())
def test_deterministic_perturbation_and_zero_change():
 m=model().eval();x,y,c,xc,u=inputs()
 a,_=m.predict_brain(x,c);b,_=m.predict_brain(x,c)
 assert torch.equal(a,b)
 delta,_=m.perturb(x,c,0,x[:,0])
 assert torch.equal(delta,torch.zeros_like(delta))
def test_mmd_properties_and_gradient():
 x=torch.randn(12,24,requires_grad=True);y=torch.randn(14,24);s=torch.tensor(48.)
 assert abs(mmd(x,x,s).item())<1e-6
 assert torch.allclose(mmd(x,y,s),mmd(y,x,s),atol=1e-6)
 assert torch.allclose(mmd(x,y,s),mmd(x[torch.randperm(12)],y,s),atol=1e-6)
 assert mmd(x,y+20,s)>mmd(x,y,s)
 mmd(x,y,s).backward();assert torch.isfinite(x.grad).all()
def test_protein_loss_off_and_condition_ablation():
 m=model(protein_weight=0,species_condition=False)
 x,y,c,xc,u=inputs();v=m.losses(x,y,c,xc,u,torch.tensor(24.))
 assert v['protein']==0 and m.condition(c)[:,:2].count_nonzero()==0
 v['loss'].backward()
 assert all(p.grad is None for p in m.protein_head.parameters())
def test_all_mapping_modes_and_representation():
 for mapping in ['ode','mlp','identity']:
  for rep in ['esm','learned']:
   m=model(mapping=mapping,representation=rep)
   x,y,c,xc,u=inputs();l=m.losses(x,y,c,xc,u,torch.tensor(24.))
   assert torch.isfinite(l['loss'])
   l['loss'].backward()

def test_condition_only_null_uses_common_noise_for_perturbation():
 m=model(source_expression=False).eval();x,y,c,xc,u=inputs()
 delta,_=m.perturb(x,c,0,0.)
 assert torch.equal(delta,torch.zeros_like(delta))

def test_every_registered_ablation_constructs_and_backpropagates():
 import json
 root=Path(__file__).resolve().parents[1]
 for path in sorted((root/'configs/ablations').glob('*.json')):
  spec=json.loads(path.read_text());m=model(**spec.get('model',{}))
  x,y,c,xc,u=inputs()
  losses=m.losses(x,y,c,xc,u,torch.tensor(24.),spec.get('shuffle_protein',False))
  assert torch.isfinite(losses['loss']),path.name
  losses['loss'].backward()
  assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None),path.name

def test_checkpoint_roundtrip_preserves_both_outputs(tmp_path):
 m=model().eval();x,y,c,xc,u=inputs()
 before=m.predict_brain(x,c)[0].detach();adt=m.predict_adt(x).detach()
 torch.save(m.state_dict(),tmp_path/'model.pt')
 n=model().eval();n.load_state_dict(torch.load(tmp_path/'model.pt',weights_only=True))
 assert torch.equal(before,n.predict_brain(x,c)[0])
 assert torch.equal(adt,n.predict_adt(x))

def test_learned_projection_has_no_sequence_initialization_prior():
 cfg=V3Config(latent_dim=8,enc_hidden=16,dec_z_hidden=16,ode_hidden=16,
              ode_n_steps=2,n_proteins=5,cond_dim=12,representation='learned')
 models=[]
 for seed in [1,2]:
  torch.manual_seed(11)
  models.append(WorldModel(np.random.default_rng(seed).normal(size=(24,12)).astype('f'),
       np.ones(24,'f'),np.zeros(5,'f'),np.ones(5,'f'),cfg).eval())
 x,y,c,xc,u=inputs()
 assert torch.equal(models[0].predict_brain(x,c)[0],models[1].predict_brain(x,c)[0])

def test_unavailable_input_coordinates_cannot_drive_prediction():
 m=model().eval();x,y,c,xc,u=inputs();mask=torch.ones(24);mask[-4:]=0
 xx=x.clone();xx[:,-4:]=999
 assert torch.equal(m.predict_brain(x,c,source_mask=mask)[0],m.predict_brain(xx,c,source_mask=mask)[0])
 assert torch.equal(m.predict_adt(x,mask),m.predict_adt(xx,mask))

def test_missing_target_coordinates_are_not_supervised_as_zero():
 m=model(stochastic=False,beta_recon=0,beta_drift=0,protein_weight=0).eval()
 x,y,c,xc,u=inputs();mask=torch.ones(24);mask[-4:]=0
 a=m.losses(x,y,c,xc,u,torch.tensor(24.),target_mask=mask)['loss']
 yy=y.clone();yy[:,-4:]=999
 b=m.losses(x,yy,c,xc,u,torch.tensor(24.),target_mask=mask)['loss']
 assert torch.equal(a,b)

def test_registered_full_models_do_not_reduce_original_scope():
 import json
 root=Path(__file__).resolve().parents[1]
 contract=json.loads((root/'docs/SCOPE_CONTRACT.json').read_text())
 protocol=json.loads((root/'configs/ablation_manifest.json').read_text())
 assert len(contract['reference_gene_names'])==25790
 for p in (root/'configs/ablations').glob('*.json'):
  spec=json.loads(p.read_text())
  assert spec['batch']==contract['reference_config']['batch_size']==1024
  assert spec['steps']==protocol['updates_by_arm'][spec['name']]
  assert spec['steps']>=protocol['minimum_formal_updates']
  assert not any(k in spec.get('model',{}) for k in ['latent_dim','esm_dim','enc_hidden','enc_depth','ode_hidden','ode_depth','ode_n_steps','dec_z_hidden','dec_z_depth'])

def test_cached_metrics_match_original_formulas_exactly():
 from evaluation import metrics,target_statistics
 p=torch.rand(16,24);y=torch.rand(13,24);mean=y.mean(0).numpy();scale=24.
 xx=(p.square().sum(1)[:,None]+p.square().sum(1)[None,:]-2*p@p.T).clamp_min(0)/scale
 yy=(y.square().sum(1)[:,None]+y.square().sum(1)[None,:]-2*y@y.T).clamp_min(0)/scale
 xy=(p.square().sum(1)[:,None]+y.square().sum(1)[None,:]-2*p@y.T).clamp_min(0)/scale
 kernel=lambda z:sum(torch.exp(-z/w) for w in [.5,1.,2.])/3
 a=metrics(p,y,mean,scale,target_statistics(y,scale));b=metrics(p,y,mean,scale)
 assert a==b
 assert a['mmd']==float(kernel(xx).mean()+kernel(yy).mean()-2*kernel(xy).mean())
 assert a['energy']==float(2*xy.sqrt().mean()-xx.sqrt().mean()-yy.sqrt().mean())

def test_zero_extra_copy_tensor_preserves_values_and_gradients():
 from data_io import tensor
 x=np.random.default_rng(9).normal(size=(8,24)).astype('f')
 old=torch.as_tensor(x.copy());new=tensor(x,'cpu')
 assert torch.equal(old,new)
 w=torch.randn(24,6,requires_grad=True)
 a=torch.autograd.grad((old@w).square().sum(),w)[0]
 b=torch.autograd.grad((new@w).square().sum(),w)[0]
 assert torch.equal(a,b)

def test_sequence_controls_preserve_missing_sequence_coverage():
 from world_model import representation_matrix
 x=np.random.default_rng(1).normal(size=(20,8)).astype('f');x[::3]=0
 covered=np.linalg.norm(x,axis=1)>0
 for kind in ['random','shuffled_esm']:
  y=representation_matrix(x,kind)
  assert np.array_equal(covered,np.linalg.norm(y,axis=1)>0)
  assert np.isfinite(y).all() and not np.array_equal(y,x)

def test_species_report_covers_multiple_celltypes_and_masks(tmp_path):
 import pandas as pd
 import scipy.sparse as sp
 from species_diagnostics import evaluate
 class Fixture:
  def __init__(self):
   self.genes=['g'+str(i) for i in range(24)]
   self.layout={'brain':['Micro','Astro'],'blood':['Mono','B']};self.pools={};self.data={};self.masks={}
   for species in ['human','mouse']:
    name=species+'_brain';assay=species+'_assay';split='test' if species=='human' else 'train'
    self.masks[assay]=np.ones(24,bool)
    self.data[name]={'x':sp.csr_matrix(np.random.default_rng(3).normal(size=(16,24)).astype('f')),'obs':pd.DataFrame(index=range(16))}
    k=0
    for ct in self.layout['brain']:
     for label in ['AD','CTRL']:
      self.pools[name,split,label,ct]={assay:{species+ct+label:np.arange(k,k+4)}};k+=4
  def routes(self,species):
   return [dict(disease=d,source_type=s,target_type=t) for d in ['AD','CTRL'] for s in ['Mono','B'] for t in ['Micro','Astro']]
  def sample(self,name,label,ct,n,rng):return rng.normal(size=(n,24)).astype('f'),np.ones(24,bool),{}
  def condition(self,n,species,label,src,tgt,device):
   c=torch.zeros(n,12,device=device);c[:,0 if species=='human' else 1]=1;c[:,2 if label=='AD' else 3]=1
   c[:,4 if src=='Mono' else 5]=1;c[:,6 if tgt=='Micro' else 7]=1;return c
 evaluate(model().eval(),Fixture(),'cpu',tmp_path)
 rows=pd.read_csv(tmp_path/'species_disease_concordance.csv')
 assert len(rows)==10 and set(rows.target_type)=={'Micro','Astro'}
 arrays=np.load(tmp_path/'species_disease_effects.npz')
 assert arrays['observed__human__Astro'].shape==(24,)
 assert arrays['predicted__mouse__B__Micro'].shape==(24,)
