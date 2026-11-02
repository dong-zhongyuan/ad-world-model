# Copied from base_supplement.R; pseudo-data creation and unused panels removed.
tpl_S6_A <- function(d, ...) {
ggplot(d,aes(cell,feature,fill=value))+geom_raster()+facet_grid(~age,scales='free_x',space='free_x')+scale_fill_viridis_c(option='magma',limits=c(0,1),name='Feature')+labs(x='Donors ordered by age',y='Encoder feature')+theme(axis.text=element_blank(),axis.ticks=element_blank(),axis.line=element_blank(),panel.spacing=grid::unit(0,'mm'))
}
tpl_S7_F <- function(d, ...) {
  # Original S7_F panel calls violin_plot after building its top-k table.
  # The caller supplies measured donor scores instead of violin_data simulations.
  violin_plot(d,'Donor program score')
}
tpl_S2_F <- function(d, ...) {
ggplot(d,aes(ratio,score,color=method,linetype=species,shape=species))+geom_line(linewidth=.45)+geom_point(size=1)+facet_wrap(~metric,ncol=2)+scale_color_manual(values=methods_col)+ylim(0,1)+labs(x='Training ratio (%)',y='Score',color=NULL)+theme(legend.position='bottom',legend.text=element_text(size=6))
}

tpl_S5_D <- function(d, ...) {
ggplot(d,aes(cell_type,gene))+geom_tile(aes(fill=expression))+geom_point(aes(size=attention),color='#222222')+scale_fill_gradient2(low='#3f99b5',mid='white',high='#ed875f',limits=c(-3,3),name='Expression z')+scale_size(range=c(.2,1.5),name='Attention')+labs(x=NULL,y=NULL)+theme(axis.text.x=element_text(angle=55,hjust=1,size=6),axis.text.y=element_text(size=5),axis.line=element_blank(),axis.ticks=element_blank(),legend.position='bottom')
}
