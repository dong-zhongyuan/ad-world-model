# Copied from base_main.R; unused paper panels and pseudo-data initializers removed.
tpl_1_A <- function(d, rings=d, ...) {
ggplot(rings,aes(x=factor(ring,levels=levels(rings$ring)),y=weight,fill=factor(category)))+geom_col(color='white',linewidth=.6,width=.9)+coord_polar(theta='y')+scale_fill_manual(values=pal)+theme_void()+theme(legend.position='none')+labs(subtitle='Species / Modality / Region / Disease / Age')
}

tpl_4_J <- function(d, special=NULL, ...) {
ggplot(d,aes(species,score,group=method,color=method))+geom_line()+geom_point()+scale_color_manual(values=methods_col)+ylim(0,1)+labs(x=NULL,y='Accuracy',color=NULL)+theme(legend.position='bottom')
}

tpl_5_E <- function(d, special=NULL, ...) {
ggplot(d,aes(cell_type,regulon,size=auc,fill=expression))+geom_point(shape=21,stroke=.3)+scale_fill_viridis_c(option='magma',direction=-1)+scale_size(range=c(.4,4))+labs(x=NULL,y=NULL,size='AUC',fill='Expression')+theme(axis.text.x=element_text(angle=50,hjust=1,size=6),axis.text.y=element_text(size=6),legend.position='right')
}

tpl_5_G <- function(d, special=NULL, ...) {
ggplot(d,aes(V1,PFC))+geom_abline(slope=1,intercept=0,linetype=2,color='#999999')+geom_point(aes(color=highlight),size=1)+ggrepel::geom_text_repel(data=special,aes(label=gene),size=2.5,seed=6674,max.overlaps=Inf)+scale_color_manual(values=c('#dddddd','#cb2638'),guide='none')+labs(x='Attention in V1',y='Attention in PFC')
}

tpl_6_G <- function(d, special=NULL, ...) {
ggplot(d,aes(predicted_age,fill=age_group))+geom_density(alpha=.7,linewidth=.35)+facet_wrap(~cell_type,ncol=3,scales='free_y')+scale_fill_manual(values=c('Young'='#80adcb','Old'='#df7b7b'))+labs(x='Predicted age (months)',y='Density',fill=NULL)+theme(legend.position='bottom',axis.text=element_text(size=6),strip.text=element_text(size=7))
}

tpl_6_I <- function(d, special=NULL, ...) {
ggplot(d)+geom_curve(aes(x=cx,y=cy,xend=x,yend=y,color=path),curvature=.15,linewidth=.35)+geom_point(aes(cx,cy,color=path),size=7)+geom_text(aes(x=x*1.11,y=y*1.11,label=gene,color=path,angle=angle*180/pi+ifelse(x<0,180,0),hjust=ifelse(x<0,1,0)),size=2.1)+scale_color_manual(values=pal)+coord_equal(xlim=c(-1.5,1.5),ylim=c(-1.5,1.5))+theme_void()+theme(legend.position='bottom')+labs(color=NULL)
}

tpl_6_J <- function(d, special=NULL, ...) {
ggplot(d,aes(cell_type,region,fill=delta))+geom_tile(color='#888888',linewidth=.2)+facet_wrap(~treatment,nrow=1)+scale_fill_gradient2(low='#248456',mid='white',high='#bda1c8',limits=c(-15,15),name='Age shift')+labs(x=NULL,y=NULL)+theme(axis.text.x=element_text(angle=90,hjust=1,size=6),axis.line=element_blank(),axis.ticks=element_blank(),legend.position='bottom')
}

tpl_7_G <- function(d, special=NULL, ...) {
ggplot(d,aes(condition,percent,fill=state))+geom_col(width=.65,color='#555555',linewidth=.3,position=position_stack(reverse=TRUE))+geom_text(aes(label=percent),position=position_stack(vjust=.5,reverse=TRUE),size=3)+scale_fill_manual(values=c('#faefbf','#bcdbb5','#efbbbb'))+labs(x=NULL,y='Percentage',fill=NULL)+theme(legend.position='bottom')
}
