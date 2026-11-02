PANELS <- list(); FIGURES <- list(); DATA_MANIFEST <- list(); PANEL_MANIFEST <- list()
pal <- c('#86bcb6','#beb7d5','#83afd0','#efaacb','#9ec866','#ef927b','#eac478','#7970ae','#c1dfaa','#659ab5','#ce79b3','#65c7ba','#a49991','#d8cf63','#bf7944','#e4bfa2','#92b6df','#d5586d','#86c4cf','#b895bc','#8ca577','#caba88','#466fa9','#ad797b','#bdcdb1','#8d8d8d','#568d80','#d6a147','#afc0e2','#cd6f5b')
methods_col <- c(scMOBA='#df8ebc',Geneformer='#9dcc60',scGPT='#67a9cf',CellWhisperer='#b39bc9',Seurat='#f89689',`Cell BLAST`='#efd774',scmap='#aaa0c8',SciBet='#aaaaaa')
classes <- c('Astro','GABA','GLU','Micro/PVM','Oligo','OPC','Vascular')
subclasses <- c('Astro','Endo','L2/3 IT','L4 IT','L5 IT','L5 ET','L5/6 NP','L6 IT','L6 IT Car3','L6 CT','L6b','LAMP5','Micro/PVM','Oligo','OPC','PVALB','SNCG','SST','VIP','VLMC')
mg <- c('MG0/1','MG2','MG3','MG4','MG5','MG6','MG7','MG8','MG10','MG11','MG12')
devtypes <- c('RG','IPC','GLU','GABA','Astro','Oligo','OPC','Micro/PVM','Vascular')
lineage <- c('EN-Non-IT-Immature','EN-L5/6 NP','EN-L5 ET','EN-L6 CT','EN-L6b','EN-IT-Immature','EN-L2/3 IT','EN-L4 IT-V1','EN-L4 IT-PFC','EN-L5 IT','EN-L6 IT')
genes <- c('BACH2','SOX5','NFIA','PBX1','SOX11','SATB2','TCF4','THRB','GLIS3','FOXP1','CUX1','POU2F1','ANK2','PLCG2','SEMA6D','GRID2','KCNIP4','RORB','GRIA4','NRXN1','CNTNAP2','NCAM1','NLGN1','SYT1','SNAP25','GABRA4','CAMK2G','PTPRD','CADM2','ROBO2','SENP7','KMT2C','AUTS2','AFF3','MLLT3','DCLK1','PDE4D','APP','CLU','GFAP','SLC1A2','CX3CR1','P2RY12','CSF1R','SRGN','SLC2A3','APOE','TREM2','LPL','SPP1','CALM1','RPS27A','UBC','GAPDH','ACTB','HSP90AA1','VIM','SLC6A1','DLG2','GRIN2A')
theme_set(theme_classic(base_size=9,base_family='sans')+theme(plot.title=element_text(size=10),axis.text=element_text(color='#333333'),legend.title=element_text(size=8),legend.text=element_text(size=7),legend.key.size=grid::unit(3,'mm'),plot.margin=margin(5,5,5,5),strip.background=element_blank(),strip.text=element_text(size=9),panel.spacing=grid::unit(3,'mm')))
seed_id <- function(id) set.seed(6674+sum(utf8ToInt(id)*seq_along(utf8ToInt(id))))
gridp <- function(...,ncol=2,rel_widths=1,rel_heights=1) plot_grid(...,ncol=ncol,rel_widths=rel_widths,rel_heights=rel_heights)
`%||%` <- function(x,y) if(is.null(x)) y else x
emb_plot <- function(d,color='cell_type',title=NULL,legend=FALSE,...) {
 ggplot(d,aes(x,y,color=.data[[color]]))+geom_point(size=.65,alpha=.95,stroke=.05)+scale_color_manual(values=if(color=='batch')c('#216ea2','#94d4dd')else pal)+coord_equal()+labs(title=title,x='UMAP-like 1',y='UMAP-like 2',color=NULL)+theme_void()+theme(plot.title=element_text(hjust=.5,size=9),legend.position=if(legend)'bottom' else 'none',legend.text=element_text(size=6)) + guides(color=guide_legend(override.aes=list(size=2),nrow=2))
}
sankey <- function(d,...) {
 ggplot(d,aes(axis1=source,axis2=target,y=count))+ggalluvial::geom_alluvium(aes(fill=source),width=.23,alpha=.65,knot.pos=.4)+ggalluvial::geom_stratum(width=.23,fill='#eeeeee',color='#444444',linewidth=.25)+geom_text(stat='stratum',aes(label=after_stat(stratum)),size=2)+scale_x_discrete(limits=c('Reference','Prediction'),expand=c(.12,.05))+scale_fill_manual(values=pal)+facet_wrap(~group,nrow=1)+labs(x=NULL,y=NULL)+theme_void()+theme(strip.text=element_text(size=9),axis.text.x=element_text(size=8),legend.position='none')
}
score_table <- function(d,...) {
 ggplot(d,aes(metric,method))+geom_point(aes(fill=value),shape=21,size=7,color='white',stroke=.4)+geom_text(aes(label=sprintf('%.2f',value)),size=2.6)+scale_fill_gradient2(low='#aa7db9',mid='#e5e5e5',high='#75bd79',midpoint=.5,limits=c(-.05,1),guide='none')+labs(x=NULL,y=NULL)+theme_minimal(base_size=9)+theme(panel.grid=element_blank(),axis.text.y=element_text(face='bold',size=8),axis.text.x=element_text(size=8))
}
violin_plot <- function(d,ylab='Score',stars=FALSE,...) {
 p<-ggplot(d,aes(group,value,fill=group))+geom_violin(linewidth=.35,scale='width')+geom_boxplot(width=.12,outlier.shape=NA,fill='white',linewidth=.3)+scale_fill_manual(values=c('#b4dcd5','#fffac6','#beb8d6','#f1b0aa'))+facet_wrap(~cohort,nrow=1)+labs(x=NULL,y=ylab)+theme(legend.position='none',axis.text.x=element_text(size=7))
 p
}
spatial <- function(d,label='Expression',black=FALSE,kind='cortex',...) {
 regions<-levels(d$region)
 p<-ggplot(d,aes(x,y,color=value))+geom_point(size=.7,stroke=.05)+coord_equal()+facet_wrap(~region,ncol=if(length(regions)==4)2 else length(regions))+labs(color=label)+theme_void()+theme(legend.position='bottom',strip.text=element_text(size=9),legend.title=element_text(size=7),legend.text=element_text(size=6),panel.spacing=grid::unit(4,'mm'))+guides(color=guide_colorbar(title.position='top',barwidth=grid::unit(25,'mm'),barheight=grid::unit(2,'mm')))
 if(black)p<-p+scale_color_gradient(low='black',high='#fff251',limits=range(d$value))+theme(panel.background=element_rect(fill='black',color=NA))else if(kind=='brain')p<-p+scale_color_viridis_c(option='plasma',limits=range(d$value))else p<-p+scale_color_gradientn(colors=c('#ffffd4','#fed98e','#f46d43','#a50026'),limits=range(d$value))
 p
}
regression <- function(d, ncol=3, ...) {
 ggplot(d,aes(actual,predicted,color=cell_type))+geom_point(size=1.1)+geom_smooth(method='lm',formula=y~x,color='#c85d5d',fill='#f3babb',linewidth=.45)+facet_wrap(~cell_type,ncol=ncol)+scale_color_manual(values=pal)+labs(x='Chronological age',y='Predicted age')+theme(legend.position='none',strip.text=element_text(size=8),panel.grid.major=element_line(color='#eeeeee',linewidth=.2))
}
