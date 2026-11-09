# -*- coding: utf-8 -*-
import bpy,json,math,pathlib,csv,os
ROOT=pathlib.Path(__file__).resolve().parents[1];data=json.loads((ROOT/'workflow_scenes.json').read_text());PROJECT=pathlib.Path(os.environ.get('AD_FIGURE_PROJECT',str(ROOT.parents[min(2,len(ROOT.parents)-1)])));ASSETS=pathlib.Path(os.environ.get('AD_WORKFLOW_ASSETS',str(PROJECT/'pd_product_figure_design/workflow_assets')));out=pathlib.Path(os.environ.get('AD_BLENDER_OUT',str(PROJECT/'delivery/ad_omics_v3/blender')));out.mkdir(parents=True,exist_ok=True);preview=pathlib.Path(os.environ.get('AD_BLENDER_PREVIEW',str(PROJECT/'delivery/ad_omics_v3/preview/blender')));preview.mkdir(parents=True,exist_ok=True)

# Every source primitive is a separately named scene object; no raster plane of the full figure.
def material(c):
 name='Color_'+c.strip('#');m=bpy.data.materials.get(name)
 if m:return m
 m=bpy.data.materials.new(name);m.use_nodes=True;nodes=m.node_tree.nodes;nodes.clear();em=nodes.new('ShaderNodeEmission');rgb=[int(c[i:i+2],16)/255 for i in (1,3,5)];lin=[v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in rgb];em.inputs['Color'].default_value=(*lin,1);em.inputs['Strength'].default_value=1;o=nodes.new('ShaderNodeOutputMaterial');m.node_tree.links.new(em.outputs[0],o.inputs['Surface']);m.diffuse_color=(*rgb,1);return m
if not os.getenv('FIGURE'):raise RuntimeError('One scene per process: set FIGURE=1..7')
for scene in data:
 if scene['figure']!=int(os.environ['FIGURE']):continue
 bpy.ops.wm.read_factory_settings(use_empty=True);bpy.context.preferences.filepaths.save_version=0
 font=bpy.data.fonts.load(os.environ.get('PD_FONT_REGULAR','/System/Library/Fonts/Supplemental/Arial.ttf'));bold=bpy.data.fonts.load(os.environ.get('PD_FONT_BOLD','/System/Library/Fonts/Supplemental/Arial Bold.ttf'))
 # Blender's font-em unit differs from typographic points. Calibrate against
 # the actual Arial H glyph metrics so 12 pt has the same physical ink size.
 font_metrics=json.loads((ROOT/'data/font_physical_metrics.json').read_text());font_scale={}
 for role,face in [('regular',font),('bold',bold)]:
  probe=bpy.data.curves.new('font_metric_probe','FONT');probe.body='H';probe.font=face;probe.size=1
  ob=bpy.data.objects.new('font_metric_probe',probe);bpy.context.scene.collection.objects.link(ob);bpy.context.view_layer.update()
  factor=font_metrics[role]['ink_width_em']/ob.dimensions.x
  assert abs(ob.dimensions.y*factor-font_metrics[role]['ink_height_em'])<.001
  font_scale[role]=factor;bpy.data.objects.remove(ob,do_unlink=True);bpy.data.curves.remove(probe)
 print('PHYSICAL_FONT_CALIBRATION',font_scale,flush=True)
 for c in list(bpy.data.collections):
  if c.name!='Collection':bpy.data.collections.remove(c)
 sc=bpy.context.scene;sc.name=f'Figure{scene["figure"]}A';w,h=scene['width'],scene['height'];groups={};inventory=[]
 def link(obj,e):
  g=e['group']
  if g not in groups:
   col=bpy.data.collections.new(g);sc.collection.children.link(col);groups[g]=col
  groups[g].objects.link(obj);obj['source_element_id']=e['id'];obj['source_type']=e['type'];obj['editable']=True
 def fillpoly(pts,e,z):
  mesh=bpy.data.meshes.new(e['id']+'_mesh');mesh.from_pydata([(x,h-y,z) for x,y in pts],[],[list(range(len(pts)))]);mesh.update();obj=bpy.data.objects.new(e['id']+'_fill',mesh);link(obj,e);obj.data.materials.append(material(e['fill']))
 def stroke(pts,e,z,closed):
  cr=bpy.data.curves.new(e['id']+'_curve','CURVE');cr.dimensions='3D';cr.resolution_u=1;cr.bevel_depth=e['lw']/2;cr.resolution_u=2;cr.bevel_resolution=2;sp=cr.splines.new('POLY');sp.points.add(len(pts)-1)
  for p,(x,y) in zip(sp.points,pts):p.co=(x,h-y,z,1)
  sp.use_cyclic_u=closed;ob=bpy.data.objects.new(e['id']+'_outline',cr);link(ob,e);ob.data.materials.append(material(e['stroke']))
 for i,e in enumerate(scene['elements']):
  z=i*6
  if e['type']=='image':
   # A separate camera-facing illustration plane; editable source geometry is in Asset_*.blend.
   import mathutils
   side=min(e['w'],e['h']);x=e['x']+(e['w']-side)/2;y=e['y']+(e['h']-side)/2
   me=bpy.data.meshes.new(e['id']+'_image_mesh');me.from_pydata([(x,h-y-side,z),(x+side,h-y-side,z),(x+side,h-y,z),(x,h-y,z)],[],[(0,1,2,3)]);me.uv_layers.new()
   for lp,uv in zip(me.uv_layers.active.data,[(0,0),(1,0),(1,1),(0,1)]):lp.uv=uv
   ob=bpy.data.objects.new(e['id']+'_illustration_'+e['asset'],me);link(ob,e)
   m=bpy.data.materials.new(e['id']+'_image_material');m.use_nodes=True;n=m.node_tree.nodes;n.clear();tex=n.new('ShaderNodeTexImage');tex.image=bpy.data.images.load(str(ASSETS/f'{e["asset"]}.png'));em=n.new('ShaderNodeEmission');tr=n.new('ShaderNodeBsdfTransparent');mix=n.new('ShaderNodeMixShader');outn=n.new('ShaderNodeOutputMaterial');m.node_tree.links.new(tex.outputs['Color'],em.inputs['Color']);m.node_tree.links.new(tex.outputs['Alpha'],mix.inputs[0]);m.node_tree.links.new(tr.outputs[0],mix.inputs[1]);m.node_tree.links.new(em.outputs[0],mix.inputs[2]);m.node_tree.links.new(mix.outputs[0],outn.inputs['Surface']);ob.data.materials.append(m)
  elif e['type']=='text':
   cr=bpy.data.curves.new(e['id']+'_type','FONT');cr.body=e['text'];cr.font=bold if e['bold'] else font;cr.size=e['size']*font_scale['bold' if e['bold'] else 'regular'];cr.align_x={'center':'CENTER','right':'RIGHT'}.get(e['align'],'LEFT');cr.align_y='CENTER';cr.space_line=1.17;cr.resolution_u=12
   obj=bpy.data.objects.new(e['id']+'_text',cr);obj.location=(e['x']+(e['w']/2 if e['align']=='center' else (e['w'] if e['align']=='right' else 0)),h-(e['y']+e['h']/2),z);link(obj,e);cr.materials.append(material(e['color']))
  else:
   if e['type']=='rect':x,y,ww,hh=e['x'],e['y'],e['w'],e['h'];pts=[(x,y),(x+ww,y),(x+ww,y+hh),(x,y+hh)];closed=True
   elif e['type']=='ellipse':pts=[(e['x']+e['w']/2+e['w']/2*math.cos(t*2*math.pi/64),e['y']+e['h']/2+e['h']/2*math.sin(t*2*math.pi/64))for t in range(64)];closed=True
   else:pts=e['pts'];closed=e['closed']
   if e['fill']!='none' and closed:fillpoly(pts,e,z)
   if e['stroke']!='none' and e['lw']>0:stroke(pts,e,z+.006,closed)
  inventory.append({k:e.get(k,'')for k in ['id','group','type','text','fill','stroke']})
 # Each biological plane retains its real editable source geometry in a hidden collection.
 for key in sorted({e['asset'] for e in scene['elements'] if e['type']=='image'}):
  with bpy.data.libraries.load(str(ASSETS/'blender'/f'Asset_{key}.blend'),link=False) as (lib,loaded):
   loaded.objects=[n for n in lib.objects if n not in ['Camera','Key','Fill','Rim']]
  col=bpy.data.collections.new('EDITABLE_SOURCE_'+key);sc.collection.children.link(col);col.hide_render=True;col.hide_viewport=True
  for ob in loaded.objects:
   if ob:col.objects.link(ob);ob['asset_source']=key
 # orthographic, pixel-consistent editable 2.5D objects
 camd=bpy.data.cameras.new('Orthographic_camera');cam=bpy.data.objects.new('Orthographic_camera',camd);sc.collection.objects.link(cam);cam.location=(w/2,h/2,5000);cam.rotation_euler=(0,0,0);camd.type='ORTHO';camd.ortho_scale=max(w,h);camd.clip_end=10000;sc.camera=cam
 sc.render.engine='CYCLES';sc.cycles.samples=8;sc.cycles.use_auto_tile=False;sc.cycles.use_denoising=False;sc.cycles.max_bounces=2;sc.render.threads_mode='FIXED';sc.render.threads=4;sc.render.resolution_x=round(w/72*600);sc.render.resolution_y=round(h/72*600);
 if hasattr(sc,'eevee') and hasattr(sc.eevee,'taa_render_samples'):sc.eevee.taa_render_samples=8
 sc.render.resolution_percentage=100;sc.render.image_settings.file_format='PNG';sc.render.film_transparent=False;sc.world=bpy.data.worlds.new('White');sc.world.color=(1,1,1);sc.world.use_nodes=True;sc.world.node_tree.nodes['Background'].inputs['Color'].default_value=(1,1,1,1);sc.view_settings.view_transform='Standard';sc.view_settings.look='None';sc.view_settings.exposure=0;sc.view_settings.gamma=1
 sc.render.filepath=str(preview/f'{str(scene["figure"])+"a" if scene["figure"]!=6 else "1h"}.png');sc['typographic_point_calibration']=json.dumps(font_scale);sc['reference_panels']=', '.join(scene['reference_panels']);sc['editable_objects']='Diagram fills, curves and text are editable objects; biological image planes map to matching editable Asset_*.blend files';sc['scientific_provenance']='Current model registry and source hashes: manuscript_review/ad_omics_v3_figures_20261031/source_lineage.json'
 bpy.ops.file.pack_all();bpy.ops.wm.save_as_mainfile(filepath=str(out/f'{str(scene["figure"])+"a" if scene["figure"]!=6 else "1h"}.blend'),compress=True);bpy.ops.render.render(write_still=True)
 with open(out/f'{str(scene["figure"])+"a" if scene["figure"]!=6 else "1h"}_elements.csv','w') as f:wr=csv.DictWriter(f,fieldnames=inventory[0].keys());wr.writeheader();wr.writerows(inventory)
 print('BLENDER_DONE',scene['figure'],len(scene['elements']),flush=True)
