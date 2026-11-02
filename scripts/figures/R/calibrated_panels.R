# Copied from calibrated_panels.R; paper fixed values and unused panels removed.
tpl_7_E <- function(d, ...) {
ggplot(d,aes(x,proportion,group=interaction(state,cohort)))+geom_line(color='#aaaaaa',linewidth=.55)+geom_point(aes(color=condition),size=2)+scale_color_manual(values=c(Normal='#3487b2',AD='#ff842a'))+scale_x_continuous(breaks=1:2,labels=c('MG4','MG8'),limits=c(.5,2.5))+scale_y_continuous(limits=c(0,21),breaks=seq(0,20,2.5),expand=c(0,0))+compact+labs(x=NULL,y='Proportions of microglial cells (%)',color=NULL)+theme(legend.position=c(.85,.87),legend.background=element_rect(fill='white',color=NA))
}

tpl_4_I <- function(d, ...) {
ggplot(d,aes(method,value,fill=method))+geom_col(width=.6)+geom_text(aes(label=sprintf('%.3f',value)),vjust=-.4,size=2)+facet_wrap(~metric,nrow=1)+scale_fill_manual(values=c('#80c8c5','#a5a0c8','#afc96b'))+scale_y_continuous(limits=c(.0,1.07),breaks=c(0,.5,1))+compact+labs(x=NULL,y=NULL)+theme(legend.position='none',axis.text.x=element_text(angle=70,hjust=1,size=6),panel.spacing=unit(2,'mm'))
}

tpl_4_G <- function(d, ...) {
ggplot(d,aes(cell_type,value,fill=cell_type))+geom_violin(scale='width',linewidth=.3)+facet_grid(gene~species)+coord_flip()+scale_fill_manual(values=pal)+compact+labs(x=NULL,y=NULL)+theme(legend.position='none',axis.text=element_text(size=6.5),strip.text=element_text(size=7),panel.spacing=unit(1.5,'mm'))
}

tpl_S1_A <- function(d, ...) {
ggplot(d,aes(species,millions,fill=modality))+geom_col(width=.8,position=position_stack(reverse=TRUE))+geom_text(aes(label=millions),position=position_stack(vjust=.5,reverse=TRUE),size=2)+scale_fill_manual(values=c('#4c72ad','#dd8850','#56a667'))+scale_y_continuous(expand=expansion(mult=c(0,.08)))+compact+labs(x=NULL,y='Cells (Million)',fill='Omics')+theme(legend.position='right',panel.grid.major.y=element_line(color='#bbbbbb',linetype='dashed',linewidth=.3))
}

enrich_native<-function(d,title=NULL,xlab='Enrichment Score',...){
 ggplot(d,aes(x,term,color=logp))+geom_point(size=2.8)+scale_color_viridis_c(name='-log10(P-value)')+compact+labs(x=xlab,y=NULL,title=title)+theme(plot.title=element_text(hjust=.5,size=8),axis.text.y=element_text(size=6),legend.title=element_text(size=6),legend.text=element_text(size=5),legend.key.height=unit(4,'mm'))
}
