if (!isTRUE(l10n_info()[['UTF-8']])) {
  Sys.setlocale('LC_CTYPE', 'en_US.UTF-8')
  stopifnot(isTRUE(l10n_info()[['UTF-8']]))
}
suppressPackageStartupMessages({library(ggplot2);library(grid);library(gtable);library(dplyr);library(tidyr);library(jsonlite)})
args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(if(length(args)) args[1] else '.')
src <- file.path(root,'source'); out <- file.path(root,'output'); dir.create(out,showWarnings=FALSE)
W <- 180
col <- c(teal='#358C80',purple='#9277B2',red='#B44D68',blue='#387FA5',ink='#27363C',gray='#ABB7BD',light='#E9EEEF')
theme_set(theme_classic(base_size=7.5,base_family='Arial')+theme(
  text=element_text(colour=col['ink']),axis.text=element_text(colour=col['ink'],size=7),
  axis.title=element_text(size=7.2),axis.title.x=element_text(margin=margin(t=3)),
  axis.title.y=element_text(margin=margin(r=3)),axis.line=element_line(linewidth=.25,colour='#718087'),
  axis.ticks=element_line(linewidth=.22,colour='#718087'),axis.ticks.length=unit(1,'mm'),
  panel.grid=element_blank(),legend.position='none',plot.margin=margin(0,0,0,0)))
records <- list(); figure <- 0; H <- 0
txt <- function(label,x,y,size=7.5,face='plain',just='left',colour=col['ink'])
  grid.text(label,x=unit(x,'mm'),y=unit(H-y,'mm'),just=c(just,'center'),gp=gpar(fontfamily='Arial',fontsize=size,fontface=face,col=colour))
panel_head <- function(letter,title,x,y) {
  # Letters follow the page grid, independently of data / axis-label margins.
  anchor <- if(figure==3) c(B=2,C=67,D=130,E=2,F=67,G=130,H=2)[[letter]] else c(B=2,C=92,D=2,E=2,F=92)[[letter]]
  txt(letter,anchor,y,10,'bold')
}
# Fix the actual ggplot panel rectangle in millimetres. Axis labels and titles
# occupy their measured natural sizes outside this rectangle, never shrink it.
measure_plot <- function(p,w,h) {
  g<-ggplotGrob(p); loc<-g$layout[g$layout$name=='panel',]
  stopifnot(nrow(loc)==1)
  g$widths[loc$l:loc$r]<-unit(w,'mm');g$heights[loc$t:loc$b]<-unit(h,'mm')
  left<-convertWidth(sum(g$widths[seq_len(loc$l-1)]),'mm',valueOnly=TRUE)
  top<-convertHeight(sum(g$heights[seq_len(loc$t-1)]),'mm',valueOnly=TRUE)
  width<-convertWidth(sum(g$widths),'mm',valueOnly=TRUE)
  height<-convertHeight(sum(g$heights),'mm',valueOnly=TRUE)
  list(g=g,left=left,top=top,width=width,height=height)
}
put <- function(p,id,x,y,w,h) {
  m<-measure_plot(p,w,h);g<-m$g;left<-m$left;top<-m$top;width<-m$width;height<-m$height
  # Center the COMPLETE graphic, including its measured axis labels.
  slot<-NULL
  if(figure==3) slot<-switch(id,B=c(0,65),E=c(0,65),C=c(65,128),F=c(65,128),D=c(128,180),G=c(128,180),H=c(0,180))
  if(figure==4 && id %in% c('B','C','E','F')) slot<-if(id %in% c('B','E'))c(0,90)else c(90,180)
  if(!is.null(slot)) x<-mean(slot)+left-width/2
  pushViewport(viewport(x=unit(x-left,'mm'),y=unit(H-y+top,'mm'),width=unit(width,'mm'),height=unit(height,'mm'),just=c('left','top')))
  grid.draw(g);popViewport()
  records[[paste(figure,id,sep='_')]]<<-list(figure=figure,panel=id,x_mm=x,top_mm=y,width_mm=w,height_mm=h,left_mm=x-left,right_mm=x-left+width,top_extent_mm=y-top,bottom_extent_mm=y-top+height,slot_mm=slot)
  invisible(x)
}
guide <- function(p,x,y,w,h) {
  g<-ggplotGrob(p+theme(legend.position='bottom',legend.direction='horizontal',legend.box='vertical',legend.title=element_text(size=6.5),legend.text=element_text(size=6.3),legend.margin=margin(0,0,0,0),legend.key.height=unit(3,'mm'),legend.key.width=unit(7,'mm')))
  k<-which(g$layout$name=='guide-box-bottom'); z<-g$grobs[[k]]
  pushViewport(viewport(x=unit(x+w/2,'mm'),y=unit(H-y-h/2,'mm'),width=unit(w,'mm'),height=unit(h,'mm')));grid.draw(z);popViewport()
}
key <- function(labels,colors,x,y,step=14,shapes=rep(16,length(labels))) {
  for(i in seq_along(labels)){
    grid.points(x=unit(x+(i-1)*step,'mm'),y=unit(H-y,'mm'),pch=shapes[i],size=unit(1.6,'mm'),gp=gpar(col=colors[i],fill=colors[i]))
    txt(labels[i],x+2+(i-1)*step,y,6.5)
  }
}
centered_key <- function(labels,colors,center,y,gap=5,shapes=rep(16,length(labels))) {
  pushViewport(viewport(gp=gpar(fontfamily='Arial',fontsize=6.5)))
  widths<-vapply(labels,function(z)convertWidth(stringWidth(z),'mm',valueOnly=TRUE),numeric(1));popViewport()
  cursor<-center-(sum(widths+2.5)+gap*(length(labels)-1))/2
  for(i in seq_along(labels)) {
    grid.points(x=unit(cursor+.8,'mm'),y=unit(H-y,'mm'),pch=shapes[i],size=unit(1.6,'mm'),gp=gpar(col=colors[i],fill=colors[i]))
    txt(labels[i],cursor+2.5,y,6.5);cursor<-cursor+2.5+widths[i]+gap
  }
}
workflow <- function(n) {
  im<-png::readPNG(file.path(src,'workflows',paste0(n,'a.png')))
  grid.raster(im,x=unit(W/2,'mm'),y=unit(H-11.7,'mm'),width=unit(W,'mm'),height=unit(W*nrow(im)/ncol(im),'mm'),interpolate=TRUE)
  # Replace the baked-in panel mark with a letter on the common page grid.
  grid.rect(x=unit(0,'mm'),y=unit(H,'mm'),width=unit(7,'mm'),height=unit(5,'mm'),just=c('left','top'),gp=gpar(fill='white',col=NA))
  txt('A',2,2.5,10,'bold')
}
gridlines <- function() theme(panel.grid.major.x=element_line(linewidth=.18,colour='#E5EAEC'))
read <- function(name,folder='action_data') read.csv(file.path(src,folder,name),check.names=FALSE,stringsAsFactors=FALSE)
targets<-c('GRN','BIN1','PILRB','BST2','ST13','ACP5','CD55','CR1','GNAS','BLNK')

draw3 <- function() {
  figure<<-3;H<<-225;grid.newpage();workflow(3)
  # One continuous coordinate system replaces two repetitive heatmaps.
  b<-read('cd4_target_coverage.csv') %>% mutate(target=factor(target_contrast_gene_name,levels=rev(targets)),condition=factor(culture_condition,levels=c('Rest','Stim8hr','Stim48hr')),row=as.numeric(target)+c(-.27,0,.27)[as.integer(condition)],magnitude=abs(ontarget_effect_size))
  condcols<-c(Rest=col['teal'],Stim8hr=col['purple'],Stim48hr=col['red']);names(condcols)<-c('Rest','Stim8hr','Stim48hr')
  pb<-ggplot(b,aes(magnitude,row))+geom_point(aes(size=n_cells_target,colour=condition),alpha=.9)+
    annotate('text',x=17,y=1,label='Not assayed',size=2.2,family='Arial',colour='#8A969B')+
    scale_colour_manual(values=condcols)+scale_size_area(max_size=2.9,breaks=c(100,600,1200),name='Cells')+
    scale_x_continuous(limits=c(-2,34),breaks=c(0,10,20,30),expand=c(0,0))+
    scale_y_continuous(limits=c(.5,10.5),breaks=1:10,labels=rev(targets),expand=c(0,0))+
    labs(x='|On-target log2FC|',y=NULL)+gridlines()+theme(axis.ticks.y=element_blank(),axis.line.y=element_blank())
  panel_head('B','CD4 actions',16,28);centered_key(c('Rest','8 h','48 h'),condcols,32.5,28,gap=5)
  put(pb,'B',16,33,46,47);guide(pb+guides(colour='none'),0,86,65,9)
  write.csv(b,file.path(out,'figure3_cd4_points.csv'),row.names=FALSE)
  # Lane-paired shifts, with connectors only within the same target and lane.
  c<-read('candidate_actions_by_lane.csv') %>% filter(target %in% c('ACP5','CR1')) %>% mutate(lane=sub('Mac_','',lane),row=ifelse(target=='ACP5',9-as.integer(sub('L','',lane)),5-as.integer(sub('L','',lane))),reduction=-on_target_delta)
  cw<-c %>% select(target,lane,row,condition,reduction) %>% pivot_wider(names_from=condition,values_from=reduction)
  pc<-ggplot(cw,aes(y=row))+geom_segment(aes(x=untreated,xend=LPS,yend=row),colour='#C4CED0',linewidth=.65)+
    geom_point(aes(x=untreated),shape=21,size=2.1,fill='white',colour=col['teal'],stroke=.6)+
    geom_point(aes(x=LPS),shape=16,size=1.8,colour=col['purple'])+
    scale_x_continuous(limits=c(0,1.2),breaks=c(0,.4,.8,1.2),expand=c(0,0))+
    scale_y_continuous(limits=c(.5,8.5),breaks=1:8,labels=c(paste('CR1',paste0('L0',4:1)),paste('ACP5',paste0('L0',4:1))),expand=c(0,0))+
    labs(x=expression(-Delta*' log expression'),y=NULL)+gridlines()+theme(axis.line.y=element_blank(),axis.ticks.y=element_blank())
  panel_head('C','Myeloid knockdown',85,28);centered_key(c('Untreated','LPS'),c(col['teal'],col['purple']),96.5,28,shapes=c(1,16));put(pc,'C',85,33,37,47)
  # Every distinct pair is shown once; diagonal self-correlations are redundant.
  cor<-as.matrix(read('blood_lps_lane_correlation.csv')[,-1]); ij<-which(upper.tri(cor),arr.ind=TRUE)
  d<-data.frame(pair=paste0('L0',ij[,1],' / L0',ij[,2]),r=cor[ij]);d$pair<-factor(d$pair,levels=rev(d$pair))
  pd<-ggplot(d,aes(r,pair))+geom_point(size=2.4,colour=col['teal'])+geom_text(aes(label=sprintf('%.2f',r)),nudge_x=.008,hjust=0,size=2.2,family='Arial')+
    scale_x_continuous(limits=c(.60,.76),breaks=c(.60,.65,.70,.75),expand=c(0,0))+labs(x='Pearson r',y=NULL)+gridlines()+theme(axis.ticks.y=element_blank(),axis.line.y=element_blank())
  panel_head('D','LPS lane agreement',144,28);put(pd,'D',144,33,32,47)
  write.csv(d,file.path(out,'figure3_lane_pairs.csv'),row.names=FALSE)
  # One matrix is retained where two-dimensional pattern is the question.
  f<-read('cd4_grn_bin1_brain_response.csv');full<-names(f)[-1];symbol<-sub(';.*','',full)
  f<-f %>% mutate(condition=1:n()) %>% pivot_longer(cols=all_of(full),names_to='members',values_to='response') %>% mutate(gene=match(members,full))
  lim<-max(abs(f$response))
  pe<-ggplot(f,aes(condition,21-gene,fill=response))+geom_tile(width=.98,height=.98)+geom_vline(xintercept=3.5,colour='white',linewidth=.7)+
    scale_fill_gradient2(low=col['blue'],mid='#FBFBFA',high=col['red'],midpoint=0,limits=c(-lim,lim),name=expression(Delta*' log expression'))+
    scale_x_continuous(limits=c(.5,6.5),breaks=1:6,labels=c('GRN\nRest','GRN\n8 h','GRN\n48 h','BIN1\nRest','BIN1\n8 h','BIN1\n48 h'),expand=c(0,0))+
    scale_y_continuous(limits=c(.5,20.5),breaks=1:20,labels=rev(symbol),expand=c(0,0))+labs(x=NULL,y=NULL)+
    theme(axis.line=element_blank(),axis.ticks=element_blank(),axis.text.x=element_text(size=6.5),axis.text.y=element_text(size=6.8))
  panel_head('E','CD4 to microglia',16,96);put(pe,'E',16,101,46,66)
  guide(pe+guides(fill=guide_colourbar(barwidth=unit(28,'mm'),barheight=unit(2,'mm'),title.position='top')),0,175,65,9)
  write.csv(data.frame(display_symbol=symbol,full_members=full),file.path(out,'figure3_gene_labels.csv'),row.names=FALSE)
  write.csv(f,file.path(out,'figure3_response_long.csv'),row.names=FALSE)
  for(item in list(list('F','acp5_microglial_response.csv','ACP5 under LPS',85,37,'Predicted microglial\nlog-expression response'),list('G','blood_lps_response.csv','Peripheral LPS input',144,32,'Measured peripheral\nlog2FC'))){
    z<-read(item[[2]]) %>% arrange(response) %>% mutate(symbol=sub(';.*','',gene),row=seq_len(n()),direction=ifelse(response>=0,'up','down'))
    pp<-ggplot(z,aes(response,row))+geom_vline(xintercept=0,linewidth=.25,colour='#78898F')+
      geom_segment(aes(x=0,xend=response,yend=row,colour=direction),linewidth=.5)+geom_point(aes(colour=direction),size=1.65)+
      scale_colour_manual(values=c(up=unname(col['red']),down=unname(col['blue'])))+scale_y_continuous(limits=c(.5,16.5),breaks=1:16,labels=z$symbol,expand=c(0,0))+
      scale_x_continuous(expand=expansion(mult=.12))+labs(x=item[[6]],y=NULL)+theme(axis.line.y=element_blank(),axis.ticks.y=element_blank(),axis.text.y=element_text(size=6.8),axis.title.x=element_text(size=6.8))
    panel_head(item[[1]],item[[3]],item[[4]],96);put(pp,item[[1]],item[[4]],101,item[[5]],66)
  }
  # A broad five-row evidence ledger grounds the figure rather than a tall footer.
  map<-read('integrated_target_map.csv');map<-map[match(targets,map$target),]; for(nm in c('cd4_perturbation_present','myeloid_perturbation_present','mr_bell_pass','mr_finn_pass')) map[[nm]]<-as.logical(map[[nm]])
  cov<-data.frame(target=targets,CD4=map$cd4_perturbation_present,Myeloid=map$myeloid_perturbation_present,Brain=!is.na(map$independent_brain_dataset)&map$independent_brain_dataset!='',AD=map$mr_bell_pass&map$mr_finn_pass,Compound=!is.na(map$compound)&map$compound!='')
  hh<-cov %>% pivot_longer(-target,names_to='evidence',values_to='present') %>% mutate(x=match(target,targets),y=6-match(evidence,c('CD4','Myeloid','Brain','AD','Compound')))
  ph<-ggplot(hh,aes(x,y))+geom_tile(fill='#F5F7F7',colour='white',linewidth=.5)+
    geom_point(data=filter(hh,present),shape=16,size=1.8,colour=col['teal'])+
    geom_text(data=filter(hh,!present),label='–',colour='#A7B2B7',size=2.3)+
    scale_x_continuous(limits=c(.5,10.5),breaks=1:10,labels=targets,expand=c(0,0))+
    scale_y_continuous(limits=c(.5,5.5),breaks=1:5,labels=c('Compound','Both AD','Brain','Myeloid','CD4'),expand=c(0,0))+
    labs(x=NULL,y=NULL)+theme(axis.line=element_blank(),axis.ticks=element_blank(),axis.text=element_text(size=7))
  panel_head('H','Evidence across ten targets',16,188);txt('Present  •     Absent  –',176,188,6.5,just='right')
  put(ph,'H',16,194,160,24);write.csv(cov,file.path(out,'figure3_evidence.csv'),row.names=FALSE)
}

draw4 <- function() {
  figure<<-4;H<<-218;grid.newpage();workflow(4)
  h<-read('headline_by_method.csv','dynamic');per<-read('combined_by_test_animal.csv','dynamic') %>% arrange(test_processing,gsm)
  s<-fromJSON(file.path(src,'dynamic/headline_metrics.json'))
  order<-c('v3_frozen_direct','world_linear','source_linear','combined_linear');hh<-h[match(order,h$method),]
  transfer<-per %>% group_by(train_processing) %>% summarise(value=100*(1-mean(mse)/mean(zero_response_mse)),.groups='drop')
  b<-data.frame(label=c('Frozen backbone','World + ridge','Source + ridge','Combined model','DMSO → inhibitors','Inhibitors → DMSO','Pooled transfer'),
    y=c(8,7,6,5,3,2,1),value=c(100*hh$relative_mse_reduction,transfer$value[match(c('dmso','inhibitors'),transfer$train_processing)],100*s$relative_mse_reduction),
    color=c(col['gray'],col['purple'],col['teal'],col['red'],col['purple'],col['teal'],col['red']))
  pb<-ggplot(b,aes(value,y))+geom_segment(aes(x=0,xend=value,yend=y),colour='#D6DEE1',linewidth=.45)+
    geom_point(aes(colour=color),size=2.2)+scale_colour_identity()+
    annotate('segment',x=100*s$animal_bootstrap_95_low,xend=100*s$animal_bootstrap_95_high,y=5,yend=5,linewidth=.5,colour=col['red'])+
    geom_text(aes(x=89,label=sprintf('%.2f%%',value)),hjust=1,size=2.15,family='Arial')+
    annotate('segment',x=0,xend=90,y=4,yend=4,linewidth=.2,colour='#DCE2E4')+
    scale_x_continuous(limits=c(0,90),breaks=c(0,20,40,60,80),expand=c(0,0))+
    scale_y_continuous(limits=c(.5,8.5),breaks=b$y,labels=b$label,expand=c(0,0))+
    labs(x='MSE reduction (%)',y=NULL)+theme(axis.line.y=element_blank(),axis.ticks.y=element_blank(),axis.text.y=element_text(size=6.5))
  panel_head('B','Model and processing comparisons',28,28);put(pb,'B',28,33,56,47)
  write.csv(b,file.path(out,'figure4_comparisons.csv'),row.names=FALSE)
  c<-bind_rows(data.frame(sample='Mean',correlation=s$mean_delta_pearson,top100=s$mean_top100_overlap),data.frame(sample=per$gsm,correlation=per$delta_pearson,top100=per$top100_overlap)) %>%
    mutate(y=5:1) %>% pivot_longer(c(correlation,top100),names_to='metric',values_to='value') %>% mutate(yy=y+ifelse(metric=='correlation',.13,-.13))
  pc<-ggplot(c,aes(value,yy,colour=metric,shape=metric))+geom_point(size=2.1)+
    geom_text(data=filter(c,sample=='Mean'),aes(label=ifelse(metric=='correlation',sprintf('r = %.3f',value),sprintf('%.2f%%',100*value))),hjust=1,nudge_x=-.035,nudge_y=ifelse(filter(c,sample=='Mean')$metric=='correlation',.15,-.15),size=2.2,family='Arial')+
    scale_colour_manual(values=c(correlation=unname(col['teal']),top100=unname(col['purple'])))+scale_shape_manual(values=c(correlation=16,top100=18))+
    scale_x_continuous(limits=c(0,1),breaks=c(0,.25,.5,.75,1),expand=c(.025,0))+
    scale_y_continuous(limits=c(.5,5.5),breaks=1:5,labels=rev(c('Mean',per$gsm)),expand=c(0,0))+
    labs(x='Score',y=NULL)+gridlines()+theme(axis.ticks.y=element_blank(),axis.line.y=element_blank(),axis.text.y=element_text(size=6.7))
  panel_head('C','Response fidelity',128,28);centered_key(c('Correlation','Top-100'),c(col['teal'],col['purple']),135,28,shapes=c(16,18));put(pc,'C',128,33,48,47)
  # Four genuinely square panels in one horizontal band, on identical scales.
  obs<-read('observed_predicted.csv','dynamic');limit<-ceiling(max(abs(c(obs$observed,obs$predicted)))*10)/10
  panel_head('D','Observed and predicted responses',28,90)
  zoom_summary<-list()
  for(i in 1:4){
    gsm<-per$gsm[i];z<-filter(obs,.data$gsm==.env$gsm)
    pp<-ggplot(z,aes(observed,predicted))+geom_abline(slope=1,intercept=0,linetype='dashed',linewidth=.25,colour='#94A4AD')+
      geom_point(size=.25,alpha=.26,colour=col['teal'],stroke=0)+
      scale_x_continuous(breaks=c(-1,0,1),expand=c(0,0))+
      scale_y_continuous(breaks=c(-1,0,1),expand=c(0,0))+
      coord_cartesian(xlim=c(-1,1),ylim=c(-1,1),expand=FALSE)+
      labs(x='Observed',y=if(i==1)'Predicted' else NULL)+theme(axis.text=element_text(size=6.4),axis.title=element_text(size=6.8))
    if(i==1) {
      first_width<-measure_plot(pp,34,34)$width
      other_width<-measure_plot(pp+labs(y=NULL),34,34)$width
      block_gap<-(180-first_width-3*other_width)/5;cursor<-block_gap
    }
    m<-measure_plot(pp,34,34);x<-cursor+m$left
    txt(gsm,x+17,92,6.5,'bold','center');txt(sprintf('r %.3f  |  %.1f%%',per$delta_pearson[i],100*per$relative_mse_reduction[i]),x+17,96,6.1,just='center')
    put(pp,paste0('D',i),x,100,34,34)
    # The inset retains all genes on the common full range. No point filtering
    # enters the statistics; the main panel only changes the viewport.
    pin<-ggplot(z,aes(observed,predicted))+
      geom_abline(slope=1,intercept=0,linetype='dashed',linewidth=.16,colour='#94A4AD')+
      geom_point(size=.12,alpha=.35,colour=col['teal'],stroke=0)+
      annotate('rect',xmin=-1,xmax=1,ymin=-1,ymax=1,fill=NA,colour='#B44D68',linewidth=.2)+
      scale_x_continuous(limits=c(-limit,limit),breaks=c(-5,0,5),expand=c(0,0))+
      scale_y_continuous(limits=c(-limit,limit),breaks=c(-5,0,5),expand=c(0,0))+
      labs(x=NULL,y=NULL)+theme(axis.text=element_text(size=3.7),axis.ticks.length=unit(.45,'mm'),axis.line=element_line(linewidth=.15),axis.ticks=element_line(linewidth=.15),plot.background=element_rect(fill='white',colour=NA))
    mi<-measure_plot(pin,10,10)
    ix<-x+.7;iy<-101
    pushViewport(viewport(x=unit(ix,'mm'),y=unit(H-iy,'mm'),width=unit(mi$width,'mm'),height=unit(mi$height,'mm'),just=c('left','top')))
    grid.draw(mi$g);popViewport()
    txt('All genes',ix+mi$width/2,100.4,4.2,just='center')
    zoom_summary[[i]]<-list(gsm=gsm,total=nrow(z),central_n=sum(abs(z$observed)<=1 & abs(z$predicted)<=1),central_fraction=mean(abs(z$observed)<=1 & abs(z$predicted)<=1),main_limits=c(-1,1),inset_limits=c(-limit,limit),metrics_use_all_genes=TRUE)
    cursor<-cursor+m$width+block_gap
  }
  write_json(zoom_summary,file.path(out,'figure4_zoom_summary.json'),pretty=TRUE,auto_unbox=TRUE,digits=12)
  # Signed paired effects use position, freeing the lower row from another heatmap.
  genes<-read('response_gene_table.csv','dynamic') %>% filter(sign(observed_mean)==sign(predicted_mean)) %>% mutate(strength=pmin(abs(observed_mean),abs(predicted_mean))) %>% arrange(desc(strength)) %>% head(18) %>% mutate(symbol=sub('\\|.*','',symbols),y=18:1)
  pe<-ggplot(genes,aes(y=y))+geom_vline(xintercept=0,linewidth=.25,colour='#A1AFB6')+
    geom_segment(aes(x=observed_mean,xend=predicted_mean,yend=y),colour='#BFCBCF',linewidth=.65)+
    geom_point(aes(x=observed_mean),shape=21,fill='white',colour=col['blue'],stroke=.5,size=1.8)+
    geom_point(aes(x=predicted_mean),colour=col['teal'],size=1.6)+
    scale_y_continuous(limits=c(.5,18.5),breaks=1:18,labels=rev(genes$symbol),expand=c(0,0))+
    scale_x_continuous(expand=expansion(mult=.08))+labs(x='Log-expression response',y=NULL)+gridlines()+theme(axis.line.y=element_blank(),axis.ticks.y=element_blank(),axis.text.y=element_text(size=6.9))
  panel_head('E','Concordant genes',28,143);centered_key(c('Observed','Predicted'),c(col['blue'],col['teal']),45,143,shapes=c(1,16));put(pe,'E',28,150,56,58)
  write.csv(genes,file.path(out,'figure4_concordant_genes.csv'),row.names=FALSE)
  terms<-read('go_biological_process_enrichment.csv','dynamic') %>% filter(query=='concordant_top100') %>% arrange(q) %>% distinct(term,.keep_all=TRUE) %>% head(8) %>% mutate(y=8:1,term_clean=sub(' \\(GO:.*','',term),label=vapply(term_clean,function(z)paste(strwrap(z,width=25),collapse='\n'),character(1)))
  pf<-ggplot(terms,aes(-log10(q),y))+geom_segment(aes(x=0,xend=-log10(q),yend=y),colour='#D6DFE1',linewidth=.5)+
    geom_point(aes(size=overlap_n),colour=col['teal'],alpha=.9)+scale_size_area(max_size=4.1,breaks=c(10,13,17),name='Genes')+
    scale_x_continuous(limits=c(0,16),breaks=c(0,5,10,15),expand=c(0,0))+
    scale_y_continuous(limits=c(.5,8.5),breaks=1:8,labels=rev(terms$label),expand=c(0,0))+
    labs(x=expression(-log[10]*' adjusted P'),y=NULL)+gridlines()+theme(axis.line.y=element_blank(),axis.ticks.y=element_blank(),axis.text.y=element_text(size=6.6,lineheight=.95))
  panel_head('F','Response programs',128,143);guide(pf,90,140.5,90,7);put(pf,'F',128,150,48,58)
  write.csv(terms,file.path(out,'figure4_programs.csv'),row.names=FALSE)
}
for(n in c(3,4)){
  height<-if(n==3)225 else 218;fun<-if(n==3)draw3 else draw4
  if(capabilities('aqua')) grDevices::quartz(file=file.path(out,paste0('Figure',n,'.pdf')),type='pdf',width=W/25.4,height=height/25.4,family='Arial') else grDevices::cairo_pdf(file.path(out,paste0('Figure',n,'.pdf')),width=W/25.4,height=height/25.4,family='Arial',onefile=FALSE)
  fun();dev.off();pdf_records<-records
  ragg::agg_png(file.path(out,paste0('Figure',n,'.png')),width=W,height=height,units='mm',res=300,background='white');fun();dev.off();records<-pdf_records
}
write_json(records,file.path(out,'panel_geometry.json'),pretty=TRUE,auto_unbox=TRUE)
writeLines(trimws(capture.output(sessionInfo()),which='right'),file.path(out,'R_sessionInfo.txt'))
cat('ggplot2 figures written to',out,'\n')
