#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
import torch,yaml
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from cadsd_jscc.deepjscc_adapter import load_deepjscc_model,deepjscc_forward_with_latents,received_latent_consistency_loss
from cadsd_jscc.metrics import psnr_per_sample
from s5_residual_refiner_pilot import build_model,gate_tensor,load_classifier,try_load_lpips
from s10_short_chain_residual_shift_diffusion import ShortChainResidualShiftDiffusion
from s13_export_coco_train2017_c8_scaleup import discover_images,select_paths,derived_seed

def R(x):
 p=Path(x);return p if p.is_absolute() else ROOT/p
def loadyaml(p): return yaml.safe_load(R(p).read_text())
def classify(model,pre,x):
 with torch.no_grad(): return model(torch.stack([pre(transforms.ToPILImage()(i.cpu())) for i in x]).to(x.device)).argmax(1)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/pc001_posterior_consistency_pilot.yaml');ap.add_argument('--device',default='cuda:0');ap.add_argument('--dry-run',action='store_true');a=ap.parse_args();c=loadyaml(a.config)
 sc=loadyaml(c['source_config']); paths=discover_images(R(sc['inputs']['source_root'])); ranked=sum(select_paths(paths,R(sc['inputs']['source_root']),sc['seed'],c['source_rank_start'],c['sample_count']),[])[c['source_rank_start']:]
 # select_paths returns first start and following count; take only following block
 if len(ranked)!=c['sample_count']: raise RuntimeError(f'posterior split mismatch {len(ranked)}')
 if a.dry_run: print(json.dumps({'count':len(ranked),'first':str(ranked[0]),'snrs':c['snrs']},indent=2));return
 out=R(c['output_dir']);
 if out.exists(): raise FileExistsError(out)
 out.mkdir(parents=True); device=torch.device(a.device); tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(256),transforms.ToTensor()])
 jscc=load_deepjscc_model(sc['baseline']['repo'],c['deepjscc_checkpoint'],8,'AWGN',c['snrs'][0],device).requires_grad_(False)
 bcfg=loadyaml(c['b1_config']);b1=build_model(bcfg).to(device);b1.load_state_dict(torch.load(R(c['b1_checkpoint']),map_location=device)['model_state_dict']);b1.eval().requires_grad_(False)
 dcfg=loadyaml(c['diffusion_config']);diff=ShortChainResidualShiftDiffusion(dcfg).to(device);diff.load_state_dict(torch.load(R(c['diffusion_checkpoint']),map_location=device)['model_state_dict']);diff.eval().requires_grad_(False)
 clf,pre,_=load_classifier(dcfg,device);lp,_=try_load_lpips(device,R('outputs/cache')); rows=[]
 for snr in map(float,c['snrs']):
  jscc.change_channel('AWGN',snr)
  for start in range(0,len(ranked),c['batch_size']):
   pp=ranked[start:start+c['batch_size']]; x=torch.stack([tf(Image.open(p).convert('RGB')) for p in pp]).to(device)
   torch.manual_seed(derived_seed(c['seed'],snr,start)); b0,_,y=deepjscc_forward_with_latents(jscc,x); sn=torch.full((len(x),),snr,device=device); norm=sn/20; bg=gate_tensor(bcfg,sn,device)
   with torch.no_grad(): anchor=b1(b0,norm,bg); raw=diff(anchor,norm,gate_tensor(dcfg,sn,device))
   before=float(received_latent_consistency_loss(jscc,raw,y)); post=raw.detach()
   for _ in range(c['proximal_steps']):
    post.requires_grad_(True); loss=received_latent_consistency_loss(jscc,post,y); g=torch.autograd.grad(loss,post)[0]; rms=g.square().flatten(1).mean(1).sqrt().clamp_min(1e-12)[:,None,None,None];post=(post-c['normalized_step_size']*g/rms).clamp(0,1).detach()
   after=float(received_latent_consistency_loss(jscc,post,y)); po=classify(clf,pre,x);pa=classify(clf,pre,anchor);pr=classify(clf,pre,raw);ppred=classify(clf,pre,post)
   with torch.no_grad(): lraw=lp(raw*2-1,x*2-1).flatten();lpost=lp(post*2-1,x*2-1).flatten();praw=psnr_per_sample(raw,x);ppsnr=psnr_per_sample(post,x)
   for i,p in enumerate(pp): rows.append({'snr_db':snr,'source':str(p.relative_to(ROOT)),'dc_before':before,'dc_after':after,'raw_psnr':float(praw[i]),'posterior_psnr':float(ppsnr[i]),'raw_lpips':float(lraw[i]),'posterior_lpips':float(lpost[i]),'anchor_correct':bool(pa[i]==po[i]),'raw_correct':bool(pr[i]==po[i]),'posterior_correct':bool(ppred[i]==po[i])})
  print('done',snr)
 fields=list(rows[0]);w=csv.DictWriter(open(out/'per_sample.csv','w',newline=''),fieldnames=fields);w.writeheader();w.writerows(rows)
 summary=[]
 for snr in map(float,c['snrs']):
  q=[r for r in rows if r['snr_db']==snr];summary.append({'snr_db':snr,'dc_delta':sum(r['dc_after']-r['dc_before'] for r in q)/len(q),'posterior_minus_raw_psnr':sum(r['posterior_psnr']-r['raw_psnr'] for r in q)/len(q),'posterior_minus_raw_lpips':sum(r['posterior_lpips']-r['raw_lpips'] for r in q)/len(q),'raw_new':sum(r['anchor_correct'] and not r['raw_correct'] for r in q),'post_new':sum(r['anchor_correct'] and not r['posterior_correct'] for r in q)})
 w=csv.DictWriter(open(out/'summary.csv','w',newline=''),fieldnames=list(summary[0]));w.writeheader();w.writerows(summary);json.dump({'config':c,'summary':summary},open(out/'metrics.json','w'),indent=2);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
