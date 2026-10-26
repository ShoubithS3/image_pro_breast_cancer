
'use strict';
const $ = id => document.getElementById(id);
const ref=$('reference'), view=$('processed'), rctx=ref.getContext('2d',{willReadFrequently:true}), vctx=view.getContext('2d',{willReadFrequently:true});
const off=document.createElement('canvas');off.width=off.height=256;const octx=off.getContext('2d',{willReadFrequently:true});
let baseline, sourceType='illustrative_phantom', sourceName='Procedural texture phantom', latestReport=null, stage=0, uploadGeneration=0;
function seededRandom(seed){return ()=>{seed=(Math.imul(1664525,seed)+1013904223)>>>0;return seed/4294967296;};}
function gaussian(rng){return Math.sqrt(-2*Math.log(Math.max(rng(),1e-9)))*Math.cos(2*Math.PI*rng());}
function stats(values){let sum=0,sq=0;for(const v of values){sum+=v;sq+=v*v;}const mean=sum/values.length;return {mean,std:Math.sqrt(Math.max(0,sq/values.length-mean*mean))};}
function setStage(value){stage=value;document.querySelectorAll('.step').forEach((el,i)=>{el.classList.toggle('active',i===stage);el.setAttribute('aria-pressed',String(i===stage));});const guides=['Start with the phantom, or choose a de-identified image. Image processing runs locally until you explicitly request server analysis.','The four texture-rich grid regions are shown below. Selection is based on texture, not disease.','Adjust blur or noise on the left. Watch the image difference and selected patches change.','Review or download the observations. The report contains no prediction or medical conclusion.'];$('stage-guide').textContent=guides[stage];}
function makePhantom(){const data=rctx.createImageData(256,256),rng=seededRandom(12);for(let y=0;y<256;y++)for(let x=0;x<256;x++){const a=x/127.5-1,b=y/127.5-1;const mask=((a+.7)**2/1.1**2+b*b/.95**2)<1;let v=mask?(.3+.13*Math.sin(25*a+10*b)+.08*Math.cos(32*b-9*a)+.25*Math.exp(-((a+.25)**2+b*b)*4)+gaussian(rng)*.035):0;v=Math.round(Math.max(0,Math.min(1,v))*255);const k=(y*256+x)*4;data.data[k]=data.data[k+1]=data.data[k+2]=v;data.data[k+3]=255;}rctx.putImageData(data,0,0);baseline=data;sourceType='illustrative_phantom';sourceName='Procedural texture phantom';$('filename').textContent='◉  '+sourceName;$('source-note').textContent='ILLUSTRATIVE PHANTOM · A procedural texture, not a mammogram or simulated cancer case.';resetControls();setStage(0);}
function resetControls(){$('blur').value=0;$('noise').value=0;render();}
function render(){if(!baseline)return;rctx.putImageData(baseline,0,0);const blur=Number($('blur').value),noise=Number($('noise').value);$('blur-value').textContent=blur.toFixed(1)+' px';$('noise-value').textContent=noise.toFixed(2);octx.clearRect(0,0,256,256);octx.filter=blur?`blur(${blur}px)`:'none';octx.drawImage(ref,0,0);octx.filter='none';const frame=octx.getImageData(0,0,256,256),rng=seededRandom(42);let diff=0,all=[];for(let i=0;i<frame.data.length;i+=4){const delta=noise?gaussian(rng)*noise*255:0;const v=Math.round(Math.max(0,Math.min(255,frame.data[i]+delta)));frame.data[i]=frame.data[i+1]=frame.data[i+2]=v;frame.data[i+3]=255;diff+=Math.abs(v-baseline.data[i])/255;all.push(v/255);}octx.putImageData(frame,0,0);vctx.putImageData(frame,0,0);let patches=[];for(let y=0;y<256;y+=64)for(let x=0;x<256;x+=64){let vals=[],nonempty=0;for(let j=y;j<y+64;j++)for(let i=x;i<x+64;i++){const v=frame.data[(j*256+i)*4]/255;vals.push(v);if(v>.05)nonempty++;}const s=stats(vals);patches.push({x,y,width:64,height:64,std:s.std,rank:nonempty/4096>.1?s.std:-1});}patches.sort((a,b)=>b.rank-a.rank);patches=patches.slice(0,4);$('patches').replaceChildren();for(let i=0;i<4;i++){const p=patches[i];vctx.strokeStyle='#d4f5a0';vctx.lineWidth=1;vctx.strokeRect(p.x+.5,p.y+.5,63,63);vctx.fillStyle='#d4f5a0';vctx.font='9px monospace';vctx.fillText('P'+(i+1),p.x+4,p.y+12);const cell=document.createElement('div'),canvas=document.createElement('canvas');canvas.width=canvas.height=64;canvas.setAttribute('aria-label','Selected texture patch '+(i+1));canvas.getContext('2d').drawImage(off,p.x,p.y,64,64,0,0,64,64);cell.append(canvas);const text=document.createElement('div');text.innerHTML=`<div class="patch-name">P${i+1} <span style="color:#8c9b82;font-size:8px">/ ${p.x},${p.y}</span></div><div class="bar"><span style="width:${Math.min(100,p.std/.5*100)}%"></span></div><div class="patch-value">σ ${p.std.toFixed(4)}</div>`;cell.append(text);$('patches').append(cell);}const imageStats=stats(all),meanChange=diff/65536;$('change').textContent=meanChange.toFixed(4);$('spread').textContent=imageStats.std.toFixed(4);$('transform-state').textContent=blur||noise?'TRANSFORMED':'CLEAN';latestReport={project:'MAMMO-LENS',demo_type:'standalone_browser_image_processing',source_type:sourceType,source_name:sourceName,working_resolution:[256,256],preprocessing:'browser grayscale conversion and resize; not the Python percentile-normalization pipeline',transformations:{gaussian_blur_radius_px:blur,gaussian_noise_std:noise,noise_seed:42},measurements:{mean_absolute_pixel_change:meanChange,intensity_std:imageStats.std},patches:patches.map(({rank,...p},i)=>({id:'P'+(i+1),...p})),trained_inference_performed:false,clinical_validation:false,limitation:'Texture patches are not lesions. No disease scores or future cancer risk estimates are produced.'};document.body.dataset.ready='true';}
$('upload').addEventListener('change',e=>{const generation=++uploadGeneration,file=e.target.files[0];$('notice').textContent='';if(!file)return;if(file.size>20*1024*1024){$('notice').textContent='Please choose an image smaller than 20 MB.';return;}if(!['image/png','image/jpeg'].includes(file.type)){$('notice').textContent='Choose a PNG or JPEG image.';return;}const image=new Image(),url=URL.createObjectURL(file);image.onload=()=>{URL.revokeObjectURL(url);if(generation!==uploadGeneration)return;if(image.naturalWidth*image.naturalHeight>40000000){$('notice').textContent='This preview supports images up to 40 megapixels. Export a smaller copy.';return;}rctx.drawImage(image,0,0,256,256);let data=rctx.getImageData(0,0,256,256);for(let i=0;i<data.data.length;i+=4){const a=data.data[i+3]/255,v=Math.round((.299*data.data[i]+.587*data.data[i+1]+.114*data.data[i+2])*a);data.data[i]=data.data[i+1]=data.data[i+2]=v;data.data[i+3]=255;}rctx.putImageData(data,0,0);baseline=data;sourceType='user_uploaded_image';sourceName=file.name;$('filename').textContent='◉  '+sourceName;$('source-note').textContent='LOCAL IMAGE · This browser sample performs image processing only. No trained disease model is loaded.';resetControls();setStage(1);};image.onerror=()=>{URL.revokeObjectURL(url);$('notice').textContent='The browser could not decode this image. Try a PNG or JPEG export.';};image.src=url;});
$('phantom').onclick=()=>{uploadGeneration++;$('upload').value='';$('notice').textContent='';makePhantom();};$('reset').onclick=resetControls;for(const id of ['blur','noise'])$(id).addEventListener('input',()=>{setStage(2);render();});document.querySelectorAll('.step').forEach(button=>button.onclick=()=>{setStage(Number(button.dataset.stage));if(stage===1)$('evidence').scrollIntoView({behavior:'smooth',block:'nearest'});if(stage===3)showReport();});function showReport(){setStage(3);$('report-json').textContent=JSON.stringify(latestReport,null,2);$('report-dialog').showModal();}$('view-report').onclick=showReport;$('close-report').onclick=()=>$('report-dialog').close();$('download').onclick=()=>{setStage(3);const blob=new Blob([JSON.stringify(latestReport,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='mammo-lens-observations.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};makePhantom();

// Connected research workflow: no clinical scores are invented client-side.
let serverResponse=null, activeRequest=0, jobTimer=null;
const initialPrediction=$('prediction-panel').innerHTML;
function invalidateServer(){activeRequest++;serverResponse=null;render();$('prediction-panel').innerHTML=initialPrediction;}
for(const id of ['blur','noise','reset','phantom','upload','model-select']){
  $(id).addEventListener(id==='blur'||id==='noise'?'input':id==='upload'||id==='model-select'?'change':'click',invalidateServer);
}
async function api(url,options={}){
  const response=await fetch(url,options);
  let body;
  try{body=await response.json();}catch{throw new Error('The server returned an unreadable response.');}
  if(!response.ok){const message=typeof body.detail==='string'?body.detail:JSON.stringify(body.detail||body);throw new Error(message);}
  return body;
}
function textNode(tag,text,className){const el=document.createElement(tag);el.textContent=text;if(className)el.className=className;return el;}
function displayNumber(value){return typeof value==='number'?value.toFixed(3):'—';}
async function refreshModels(){
  const rows=await api('/api/models'),select=$('model-select'),previous=select.value;
  select.replaceChildren(new Option('Image processing only',''));
  rows.forEach(row=>{const option=new Option(`${row.id} · ${row.kind}/${row.mode}${row.provenance==='synthetic-smoke'?' · SYNTHETIC':''}`,row.id);option.disabled=row.provenance==='synthetic-smoke';select.add(option);});
  if(rows.some(r=>r.id===previous))select.value=previous;
}
function showPrediction(result){
  const panel=$('prediction-panel');panel.replaceChildren();
  if(!result.prediction){panel.append(textNode('div','Image processing complete. No trained model was selected.','model-state'));return;}
  const p=result.prediction;
  panel.append(textNode('div',`${p.model_id} · ${p.kind} / ${p.mode}`,'small'));
  panel.append(textNode('div',displayNumber(p.score),'score'));
  panel.append(textNode('div',`Reference ${displayNumber(p.reference_score)}  /  Score shift ${p.score_shift>=0?'+':''}${displayNumber(p.score_shift)}`,'comparison'));
  panel.append(textNode('p','Experimental malignant-class score; not future cancer risk or a validated clinical probability.','small'));
  panel.append(textNode('div',p.low_confidence?'Exploratory low-confidence flag: flagged.':'Exploratory low-confidence flag: not flagged. A high score does not establish reliability.','alert'));
  panel.append(textNode('p','Flag threshold 0.70 is unvalidated. Do not use this output for clinical decisions.','small'));
  if(p.density_probabilities){
    panel.append(textNode('p',`Auxiliary density head: ${p.density_probabilities.map((v,i)=>`${i+1}: ${(v*100).toFixed(1)}%`).join(' · ')}`,'small'));
  }
  if(p.patch_masking.length){
    const title=textNode('p','Patch-masking response / replace one region with image-median intensity','small');panel.append(title);
    p.patch_masking.forEach((row,i)=>panel.append(textNode('div',`P${i+1} → score ${displayNumber(row.score)} / Δ ${row.delta>=0?'+':''}${displayNumber(row.delta)}`,'small')));
    panel.append(textNode('p','Patch selection is recomputed after masking. This measures the full pipeline response, not causal explanation or lesion significance.','small'));
  }
}
function drawApiImage(canvas,url){return new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>{canvas.getContext('2d').drawImage(img,0,0,256,256);resolve();};img.onerror=reject;img.src=url;});}
$('analyze').onclick=async()=>{
  const file=$('upload').files[0];
  if(sourceType!=='user_uploaded_image'||!file||sourceName!==file.name){$('api-status').textContent='Choose a valid de-identified image first. The illustrative phantom is not scored.';return;}
  const request=++activeRequest,button=$('analyze');button.disabled=true;$('api-status').textContent='Analyzing with the Python pipeline…';
  const form=new FormData();form.append('image',file);form.append('model_id',$('model-select').value);form.append('blur',$('blur').value);form.append('noise',$('noise').value);form.append('mask_patches','true');
  try{
    const result=await api('/api/analyze',{method:'POST',body:form});
    if(request!==activeRequest)return;
    serverResponse=result;latestReport={project:'MAMMO-LENS',source_type:'user_uploaded_image',source_name:file.name,...result};delete latestReport.reference_image;delete latestReport.processed_image;
    await Promise.all([drawApiImage(ref,result.reference_image),drawApiImage(view,result.processed_image)]);
    $('change').textContent=result.measurements.mean_absolute_pixel_change.toFixed(4);$('spread').textContent=result.measurements.intensity_std.toFixed(4);
    // Rebuild previews from the server-selected boxes rather than stale browser selections.
    const image=new Image();image.src=result.processed_image;await image.decode();
    const patches=$('patches');patches.replaceChildren();
    result.patches.forEach(p=>{const cell=document.createElement('div'),canvas=document.createElement('canvas');canvas.width=canvas.height=64;const [x,y,x1,y1]=p.box;canvas.getContext('2d').drawImage(image,x,y,x1-x,y1-y,0,0,64,64);cell.append(canvas,textNode('div',p.id,'patch-name'),textNode('div',`σ ${p.texture_std.toFixed(4)}`,'patch-value'));patches.append(cell);});
    showPrediction(result);$('api-status').textContent='Server analysis complete · Python percentile normalization.';
  }catch(error){if(request===activeRequest)$('api-status').textContent=error.message;}finally{button.disabled=false;}
};
async function refreshExperiments(){
  const target=$('experiment-table');
  try{
    const rows=await api('/api/experiments');target.replaceChildren();
    if(!rows.length){target.append(textNode('p','No completed runs yet. Train with a real dataset or use the clearly labelled synthetic smoke command to test the pipeline.','small'));return;}
    const table=document.createElement('table'),head=document.createElement('tr');['Run','Model','Source','AUROC','Sensitivity','Specificity','Brier','Details'].forEach(label=>head.append(textNode('th',label)));table.append(head);
    rows.forEach(row=>{
      const report=row.report,metrics=report.splits?.test?.metrics||{},tr=document.createElement('tr');
      [row.id,`${row.kind}/${row.mode}`,row.provenance,...['auroc','sensitivity','specificity','brier'].map(key=>displayNumber(metrics[key]))].forEach(value=>tr.append(textNode('td',value)));
      const td=document.createElement('td'),button=textNode('button','Inspect','text-btn');button.onclick=()=>{const detail=$('experiment-detail');detail.replaceChildren(textNode('pre',JSON.stringify(report,null,2)));};td.append(button);tr.append(td);table.append(tr);
    });target.append(table);
  }catch(error){target.textContent=error.message;}
}
function operatorHeaders(){return {'Authorization':'Bearer '+$('admin-token').value,'Content-Type':'application/json'};}
$('audit-data').onclick=async()=>{try{const data=await api('/api/dataset',{headers:operatorHeaders()});$('dataset-summary').hidden=false;$('dataset-summary').textContent=JSON.stringify(data,null,2);$('training-notice').textContent='Dataset audit completed.';}catch(error){$('training-notice').textContent=error.message;}};
async function refreshJobs(){
  try{
    const rows=await api('/api/jobs',{headers:operatorHeaders()}),target=$('job-list');target.replaceChildren();
    rows.forEach(row=>{const div=textNode('div',`${row.id} / ${row.status} / ${row.request.kind} ${row.request.mode} `,'job'),button=textNode('button','View log','text-btn');button.onclick=async()=>{try{const job=await api('/api/jobs/'+row.id,{headers:operatorHeaders()});$('job-log').hidden=false;$('job-log').textContent=job.log_tail||'Waiting for training output…';}catch(error){$('training-notice').textContent=error.message;}};div.append(button);target.append(div);});
    const running=rows.some(r=>r.status==='queued'||r.status==='running');
    if(running&&!jobTimer)jobTimer=setInterval(refreshJobs,5000);
    if(!running&&jobTimer){clearInterval(jobTimer);jobTimer=null;await refreshModels();await refreshExperiments();}
  }catch(error){$('training-notice').textContent=error.message;if(jobTimer){clearInterval(jobTimer);jobTimer=null;}}
}
$('start-training').onclick=async()=>{
  const button=$('start-training');button.disabled=true;
  try{
    const request={kind:$('train-kind').value,mode:$('train-mode').value,epochs:Number($('train-epochs').value),batch_size:Number($('train-batch').value),seed:42,density_aware:$('train-density').value==='true',fine_tune:false};
    const job=await api('/api/jobs',{method:'POST',headers:operatorHeaders(),body:JSON.stringify(request)});
    $('training-notice').textContent=`Queued ${job.id}. Keep the server running; progress is recorded locally.`;await refreshJobs();
  }catch(error){$('training-notice').textContent=error.message;}finally{button.disabled=false;}
};
$('refresh-jobs').onclick=refreshJobs;$('refresh-experiments').onclick=async()=>{await refreshModels();await refreshExperiments();};
(async()=>{try{const health=await api('/api/health');$('api-status').textContent=`API connected · ${health.models} completed models · ${health.dataset_available?'manifest found':'dataset manifest needed'}.`;await refreshModels();await refreshExperiments();}catch(error){$('api-status').textContent='API unavailable. Start the documented FastAPI server to enable training and inference.';}})();
