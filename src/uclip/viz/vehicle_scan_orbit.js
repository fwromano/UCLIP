'use strict';
const D=JSON.parse(document.getElementById('orbit-data').textContent),$=id=>document.getElementById(id);
const categories=Object.keys(D.colors),fullGroups=Object.fromEntries(categories.map(c=>[c,D.rows.filter(r=>r.category===c)]));
const params=new URLSearchParams(location.search);
const nearestElevation=value=>D.elevations.reduce((best,v)=>Math.abs(v-value)<Math.abs(best-value)?v:best,D.elevations[0]);
let requestedElevation=Number(params.get('elevation')??54);
if(!Number.isFinite(requestedElevation))requestedElevation=54;
requestedElevation=Math.max(D.elevations[0],Math.min(D.elevations.at(-1),requestedElevation));
let elevation=nearestElevation(requestedElevation);
let groups,lookup,representatives;
function rebuildViews(){groups=Object.fromEntries(categories.map(c=>[c,fullGroups[c].filter(r=>r.elevation===null||r.elevation===elevation)]));lookup=Object.fromEntries(categories.map(c=>[c,Array.from({length:360},(_,a)=>nearest(c,a))]));representatives=Object.fromEntries(categories.map(c=>[c,[...new Map(groups[c].map(r=>[r.heading,r])).values()].sort((a,b)=>a.heading-b.heading)]));}
const byRow=new Map(D.rows.map(r=>[r.embedding_row,r]));
const byScanRow=new Map(D.rows.filter(r=>r.dataset==='scans').map(r=>[r.source_embedding_row,r]));
let selectedCameraSample=null;

const circular=(a,b)=>Math.abs((a-b+540)%360-180);
const nearest=(c,a)=>groups[c].reduce((best,r)=>circular(r.heading,a)<circular(best.heading,a)?r:best,groups[c][0]);

rebuildViews();
const ranges=[0,1,2].map(i=>{const v=[...D.rows,...D.cameraSamples].map(r=>r.coords[i]),lo=Math.min(...v),hi=Math.max(...v);return[lo,hi]});

let color=categories.includes(params.get('color'))?params.get('color'):D.defaultSeries;
const visibleSeries=new Set(params.has('show')?params.getAll('show').filter(c=>categories.includes(c)):categories);
let allElevations=params.get('elevations')!=='current',showPaths=params.get('paths')==='1';
let colorBy=params.get('colorBy')==='series'?'series':'viewpoint';
// One mapping drives the matrix, each measured sample and its PCA point.
// Editable colors at headings 0/180 and elevations -5/90; a continuous Oklab blend preserves the four endpoints.
const defaultPalette='red-blue-green-yellow';
const viewPalettes={
 'red-blue-green-yellow':{label:'Red / blue / green / yellow',names:{topLeft:'green',topRight:'red',bottomLeft:'blue',bottomRight:'yellow'},anchors:{topLeft:[0,166,81],topRight:[230,57,70],bottomLeft:[38,98,255],bottomRight:[255,212,0]}},
 'teal-violet':{label:'Teal × violet',names:{topLeft:'teal',topRight:'pale blue',bottomLeft:'slate',bottomRight:'violet'},anchors:{topLeft:[95,208,184],topRight:[218,227,237],bottomLeft:[101,113,131],bottomRight:[180,160,229]}},
 'blue-amber':{label:'Blue × amber',names:{topLeft:'blue',topRight:'ivory',bottomLeft:'slate',bottomRight:'amber'},anchors:{topLeft:[104,174,223],topRight:[240,226,200],bottomLeft:[101,113,131],bottomRight:[228,181,108]}},
 'custom-corners':{label:'Green / red / orange / black',names:{topLeft:'green',topRight:'red',bottomLeft:'orange',bottomRight:'black'},anchors:{topLeft:[0,166,81],topRight:[230,57,70],bottomLeft:[245,140,35],bottomRight:[0,0,0]}}
};
const anchorNames=['bottomLeft','topLeft','topRight','bottomRight'];
const rgbHex=rgb=>'#'+rgb.map(v=>v.toString(16).padStart(2,'0')).join('');
const hexRgb=hex=>[1,3,5].map(i=>parseInt(hex.slice(i,i+2),16));
const validHex=value=>typeof value==='string'&&/^#[0-9a-f]{6}$/i.test(value);
let savedAnchors=null;
try{savedAnchors=JSON.parse(localStorage.getItem('vehicle-view-hsv-colors'))}catch{}
const customAnchors=Object.fromEntries(anchorNames.map(name=>{
 const url=params.get('anchor-'+name),saved=savedAnchors?.[name];
 return [name,validHex(url)?hexRgb(url):validHex(saved)?hexRgb(saved):[...viewPalettes[defaultPalette].anchors[name]]];
}));
viewPalettes.custom={label:'My colors · smooth blend',anchors:customAnchors};
let palette=Object.hasOwn(viewPalettes,params.get('palette'))?params.get('palette'):
 (anchorNames.some(name=>params.has('anchor-'+name))?'custom':defaultPalette);
let viewAnchors=viewPalettes[palette].anchors;
// Oklab matrices: https://bottosson.github.io/posts/oklab/ (public-domain reference).
// Cartesian blending avoids hue-arc direction flips inside a four-corner grid.
function toOklab(rgb){
 const [r,g,b]=rgb.map(v=>{const c=v/255;return c<=.04045?c/12.92:((c+.055)/1.055)**2.4});
 const l=Math.cbrt(.4122214708*r+.5363325363*g+.0514459929*b);
 const m=Math.cbrt(.2119034982*r+.6806995451*g+.1073969566*b);
 const s=Math.cbrt(.0883024619*r+.2817188376*g+.6299787005*b);
 return [.2104542553*l+.793617785*m-.0040720468*s,1.9779984951*l-2.428592205*m+.4505937099*s,.0259040371*l+.7827717662*m-.808675766*s];
}
function fromOklab([L,a,b]){
 const l=(L+.3963377774*a+.2158037573*b)**3;
 const m=(L-.1055613458*a-.0638541728*b)**3;
 const s=(L-.0894841775*a-1.291485548*b)**3;
 return [4.0767416621*l-3.3077115913*m+.2309699292*s,-1.2684380046*l+2.6097574011*m-.3413193965*s,-.0041960863*l-.7034186147*m+1.707614701*s].map(v=>{
  const c=Math.max(0,Math.min(1,v));return Math.round(255*(c<=.0031308?12.92*c:1.055*c**(1/2.4)-.055));
 });
}
let viewLab=Object.fromEntries(Object.entries(viewAnchors).map(([name,rgb])=>[name,toOklab(rgb)]));
function viewpointColor(heading,elevation){
 const h=((heading%360)+360)%360,x=(h<=180?h:360-h)/180;
 const y=elevation===null?.5:Math.max(0,Math.min(1,(elevation-D.elevations[0])/(D.elevations.at(-1)-D.elevations[0])));
 const weights={bottomLeft:(1-x)*(1-y),bottomRight:x*(1-y),topLeft:(1-x)*y,topRight:x*y};
 const lab=[0,1,2].map(i=>anchorNames.reduce((sum,name)=>sum+weights[name]*viewLab[name][i],0));
 return `rgb(${fromOklab(lab).join(' ')})`;
}
let viewpointColors=new Map(D.rows.map(row=>[row.embedding_row,viewpointColor(row.heading,row.elevation)]));
const pointColor=row=>colorBy==='viewpoint'?viewpointColors.get(row.embedding_row):D.colors[row.category];
const seriesCheckboxes=new Map();
const fullRings=Object.fromEntries(categories.map(c=>{const rings=new Map();for(const r of fullGroups[c]){if(!rings.has(r.elevation))rings.set(r.elevation,new Map());if(!rings.get(r.elevation).has(r.heading))rings.get(r.elevation).set(r.heading,r)}return[c,[...rings.values()].map(r=>[...r.values()].sort((a,b)=>a.heading-b.heading))]}));
let angle=Math.round(Math.max(0,Math.min(360,Number(params.get('heading'))||0)));
let mode=['3d','01','02','12'].includes(params.get('projection'))?params.get('projection'):'3d';
let selected=null,playing=false,continuous=angle,lastFrame=0,lastUpdate=0,urlTimer=0;
let yaw=-.65,pitch=.55,zoom=1,plotWidth=0,plotHeight=0,screenPoints=[],staticFrame=0;
let pendingRow=null,displayBusy=false,prefetchBusy=false,activeSheet=null;
const sheets=new Map(),loading=new Map(),MAX_SHEETS=3;
const plot=$('plot'),base=document.createElement('canvas'),overlay=document.createElement('canvas'),tooltip=document.createElement('div');
base.setAttribute('aria-hidden','true');overlay.setAttribute('role','img');overlay.setAttribute('aria-label','PCA projection; drag to rotate and click a measured view');tooltip.className='plot-tooltip';plot.append(base,overlay,tooltip);
const ctx=base.getContext('2d'),marker=overlay.getContext('2d'),preview=$('jeep-preview').getContext('2d');
const coordinateValues=[];
for(let i=0;i<3;i++){const e=document.createElement('div');e.className='coordinate';const b=document.createElement('b');b.textContent=`PC${i+1}`;const span=document.createElement('span');e.append(b,span);$('coordinates').append(e);coordinateValues.push(span);}
function syncUrl(){clearTimeout(urlTimer);urlTimer=setTimeout(()=>{const p=new URLSearchParams({color,heading:String(angle),elevation:String(requestedElevation),projection:mode,elevations:allElevations?'all':'current',paths:showPaths?'1':'0',colorBy,palette});if(palette==='custom')for(const name of anchorNames)p.set('anchor-'+name,rgbHex(viewAnchors[name]));for(const c of categories)if(visibleSeries.has(c))p.append('show',c);if(!visibleSeries.size)p.append('show','');history.replaceState(null,'','?'+p)},150);}
function state(){window.orbitState={heading:angle,elevation,requestedElevation,visibleSeries:[...visibleSeries],allElevations,showPaths,colorBy,palette,anchors:Object.fromEntries(anchorNames.map(name=>[name,rgbHex(viewAnchors[name])])),interpolation:'OKLab',selectedColor:selected?pointColor(selected):null,selectedElevation:selected?.elevation??null,color,mode,selectedRow:selected?.embedding_row??null,selectedHeading:selected?.heading??null,coords:selected?[...selected.coords]:[],playing,markerRow:selected?.embedding_row??null,imageRow:selected?.embedding_row??null,pendingRow:pendingRow?.embedding_row??null,previewSheets:sheets.size};}
function evict(){while(sheets.size>MAX_SHEETS){const key=[...sheets.keys()].find(k=>k!==activeSheet);if(key===undefined)break;sheets.delete(key)}}
async function loadSheet(path){if(sheets.has(path)){const img=sheets.get(path);sheets.delete(path);sheets.set(path,img);return img}if(loading.has(path))return loading.get(path);
 const job=(async()=>{const img=new Image();img.src=D.atlases?.[path]||path;await img.decode();sheets.set(path,img);evict();return img})();loading.set(path,job);try{return await job}finally{loading.delete(path)}}
function prefetch(){if(!playing||prefetchBusy||!selected)return;const next=lookup[color][(angle+15)%360];if(next.preview_sheet===selected.preview_sheet||sheets.has(next.preview_sheet))return;prefetchBusy=true;loadSheet(next.preview_sheet).catch(()=>{}).finally(()=>{prefetchBusy=false})}
function project(coords){if(mode!=='3d'){const ax=[+mode[0],+mode[1]],pad=52;return{x:pad+(coords[ax[0]]-ranges[ax[0]][0])/(ranges[ax[0]][1]-ranges[ax[0]][0])*(plotWidth-pad*2),y:plotHeight-pad-(coords[ax[1]]-ranges[ax[1]][0])/(ranges[ax[1]][1]-ranges[ax[1]][0])*(plotHeight-pad*2),depth:0}}
 const n=coords.map((v,i)=>(v-(ranges[i][0]+ranges[i][1])/2)/((ranges[i][1]-ranges[i][0])/2));const x=Math.cos(yaw)*n[0]-Math.sin(yaw)*n[1],y=Math.sin(yaw)*n[0]+Math.cos(yaw)*n[1];const z=Math.sin(pitch)*y+Math.cos(pitch)*n[2],depth=Math.cos(pitch)*y-Math.sin(pitch)*n[2],scale=Math.min(plotWidth*.28,plotHeight*.30)*zoom;return{x:plotWidth/2+x*scale,y:plotHeight/2-z*scale,depth};}
function line(a,b,color,width=1,dashed=false){ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dashed?[4,5]:[]);ctx.stroke();ctx.setLineDash([])}
function axes(){ctx.font='11px system-ui';ctx.fillStyle='#a4b3c9';if(mode!=='3d'){const ax=[+mode[0],+mode[1]],pad=52;for(let i=0;i<=4;i++){const x=pad+(plotWidth-pad*2)*i/4,y=plotHeight-pad-(plotHeight-pad*2)*i/4;line({x,y:pad},{x,y:plotHeight-pad},'#263449');line({x:pad,y},{x:plotWidth-pad,y},'#263449');ctx.fillText((ranges[ax[0]][0]+(ranges[ax[0]][1]-ranges[ax[0]][0])*i/4).toFixed(2),x-12,plotHeight-pad+18);ctx.fillText((ranges[ax[1]][0]+(ranges[ax[1]][1]-ranges[ax[1]][0])*i/4).toFixed(2),8,y+4)}ctx.textAlign='center';ctx.fillText(`PC${ax[0]+1} · ${(D.ratio[ax[0]]*100).toFixed(1)}%`,plotWidth/2,plotHeight-10);ctx.save();ctx.translate(11,plotHeight/2);ctx.rotate(-Math.PI/2);ctx.fillText(`PC${ax[1]+1} · ${(D.ratio[ax[1]]*100).toFixed(1)}%`,0,0);ctx.restore();ctx.textAlign='left';return}
 const origin=ranges.map(r=>r[0]);for(let axis=0;axis<3;axis++){const end=[...origin];end[axis]=ranges[axis][1];line(project(origin),project(end),'#50627f');const p=project(end);ctx.fillText(`PC${axis+1} · ${(D.ratio[axis]*100).toFixed(1)}%`,p.x+4,p.y+4);for(let step=0;step<=4;step++){for(let other=axis+1;other<3;other++){const a=[...origin],b=[...origin];a[axis]=b[axis]=ranges[axis][0]+(ranges[axis][1]-ranges[axis][0])*step/4;b[other]=ranges[other][1];line(project(a),project(b),'#263449')}}}}
function staticPlot(){
 staticFrame=0;if(!plotWidth)return;ctx.clearRect(0,0,plotWidth,plotHeight);axes();screenPoints=[];
 const counts={},plottedElevations={};
 for(const c of categories){if(!visibleSeries.has(c))continue;const active=c===color,hex=D.colors[c];
  const points=allElevations?fullGroups[c]:groups[c],rings=allElevations?fullRings[c]:[representatives[c]];
  if(showPaths){ctx.globalAlpha=active?.65:.45;for(const ring of rings){for(let i=1;i<ring.length;i++)line(project(ring[i-1].coords),project(ring[i].coords),pointColor(ring[i]),1,ring[i].heading-ring[i-1].heading>(D.seriesMeta[c].step??5));if(ring.length>1)line(project(ring.at(-1).coords),project(ring[0].coords),pointColor(ring[0]),1,true)}}
  ctx.globalAlpha=colorBy==='viewpoint'?1:active?.8:.6;
  for(const row of points){const p=project(row.coords),fill=pointColor(row);ctx.fillStyle=fill;ctx.beginPath();ctx.arc(p.x,p.y,active?3:2.5,0,Math.PI*2);ctx.fill();if(colorBy==='viewpoint'){ctx.strokeStyle='#a4b3c9';ctx.lineWidth=.6;ctx.stroke()}screenPoints.push({...p,row,fill})}
  counts[c]=points.length;plottedElevations[c]=[...new Set(points.map(r=>r.elevation))];ctx.globalAlpha=1;
 }
 if(visibleSeries.has(color)){const front=byRow.get(D.references[color]);if(allElevations||front.elevation===null||front.elevation===elevation){const f=project(front.coords);ctx.fillStyle='#a4b3c9';ctx.fillRect(f.x-3,f.y-3,6,6)}}
 window.orbitPlotState={series:categories.filter(c=>visibleSeries.has(c)),counts,elevations:plottedElevations,rows:screenPoints.length,rowIds:screenPoints.map(p=>p.row.embedding_row),allElevations,showPaths,colorBy,palette};
 $('visible-count').textContent=`· ${visibleSeries.size} selected · ${screenPoints.length.toLocaleString()} views`;
 for(const [c,entry] of seriesCheckboxes){entry.count.textContent=(allElevations?fullGroups[c]:groups[c]).length;entry.label.dataset.inspected=String(c===color)}
 $('color-key').textContent=colorBy==='viewpoint'?'PCA dots match the HSV grid: 0° → 180° → 360°; 360° repeats 0°. Edit the four swatches below.':'PCA dots show vehicle paint; the matrix still shows viewpoint colors.';drawMarker();
}
function scheduleStatic(){if(!staticFrame)staticFrame=requestAnimationFrame(staticPlot)}
function drawMarker(){
 marker.clearRect(0,0,plotWidth,plotHeight);window.orbitPoint=null;window.cameraOrbitPoint=null;if(!plotWidth)return;
 if(selected&&visibleSeries.has(selected.category)&&(allElevations||selected.elevation===null||selected.elevation===elevation)){
  const p=project(selected.coords);marker.beginPath();marker.arc(p.x,p.y,8,0,Math.PI*2);marker.fillStyle=pointColor(selected);marker.fill();marker.lineWidth=2;marker.strokeStyle='#fff';marker.stroke();marker.font='12px system-ui';marker.textAlign='center';marker.fillStyle='#fff';marker.fillText(`${selected.heading}° / ${selected.elevation??'?'}°`,p.x,p.y-14);window.orbitPoint={x:p.x,y:p.y,row:selected.embedding_row,fill:pointColor(selected)};
 }
 if(!selectedCameraSample)return;const native=byScanRow.get(selectedCameraSample.selected_embedding_row);
 if(!native||!visibleSeries.has(native.category)||(!allElevations&&native.elevation!==elevation))return;
 const q=project(selectedCameraSample.coords),r=project(native.coords);marker.beginPath();marker.moveTo(r.x,r.y);marker.lineTo(q.x,q.y);marker.strokeStyle='#4ef0cb80';marker.lineWidth=1;marker.stroke();marker.beginPath();marker.arc(r.x,r.y,4,0,Math.PI*2);marker.strokeStyle='#4ef0cb';marker.stroke();marker.beginPath();marker.moveTo(q.x,q.y-7);marker.lineTo(q.x+7,q.y);marker.lineTo(q.x,q.y+7);marker.lineTo(q.x-7,q.y);marker.closePath();marker.fillStyle='#4ef0cb';marker.fill();marker.strokeStyle='#fff';marker.lineWidth=1.5;marker.stroke();marker.fillStyle='#b8ffeb';marker.fillText(selectedCameraSample.observer+' crop',q.x,q.y+22);window.cameraOrbitPoint={x:q.x,y:q.y,sourceRow:selectedCameraSample.selected_embedding_row,cameraRow:selectedCameraSample.embedding_row,coords:selectedCameraSample.coords};
}
function resize(){const rect=plot.getBoundingClientRect(),dpr=devicePixelRatio||1;plotWidth=rect.width;plotHeight=rect.height;for(const [canvas,context] of [[base,ctx],[overlay,marker]]){canvas.width=Math.round(plotWidth*dpr);canvas.height=Math.round(plotHeight*dpr);context.setTransform(dpr,0,0,dpr,0,0)}staticPlot()}
function showRow(row,image){activeSheet=row.preview_sheet;selected=row;const cell=row.preview_cell;preview.clearRect(0,0,256,176);preview.drawImage(image,(cell%8)*256,Math.floor(cell/8)*176,256,176,0,0,256,176);$('jeep-preview').dataset.row=row.embedding_row;$('jeep-preview').setAttribute('aria-label',`${row.category}, relative heading ${row.heading} degrees, elevation ${row.elevation??'unknown'}`);
 updateViewPad();const mismatch=circular(row.heading,angle);$('selected-view').textContent=`${row.category} · ${row.heading}° / ${row.elevation??'?'}°`;updateMismatch(row);$('image-name').textContent=row.image_path.split('/').at(-1);$('row-info').textContent=`Source cache row ${row.source_embedding_row} · 512 values`;row.coords.forEach((v,i)=>coordinateValues[i].textContent=v.toFixed(4));const ref=byRow.get(D.references[row.category]);$('reference-label').textContent=`Cosine to reference (${ref.heading}° / ${ref.elevation??'?'}°)`;$('similarity').textContent=row.cos_to_front.toFixed(4);$('view-note').textContent=row.uncertainty?`Source angle has ±${row.uncertainty}° label uncertainty.`:(row.dataset==='sim2'?'Sim2: verified heading, uncalibrated elevation.':'Scans: inferred heading convention; nearest recorded view. No interpolation.');$('original').hidden=row.dataset!=='sim2';$('original').style.display=row.dataset==='sim2'?'inline-block':'none';$('original').href=D.imageBase+row.image_path;$('status').textContent='';drawMarker();state();evict();prefetch();}
async function requestDisplay(row){pendingRow=row;if(displayBusy)return;displayBusy=true;try{while(pendingRow){const current=pendingRow;pendingRow=null;if(selected?.embedding_row===current.embedding_row){updateMismatch(current);updateViewPad();state();prefetch();continue}$('status').textContent=sheets.has(current.preview_sheet)?'':'Loading preview…';const image=await loadSheet(current.preview_sheet);if(pendingRow&&pendingRow.embedding_row!==current.embedding_row)continue;showRow(current,image)}}catch(error){stop();$('status').textContent='Preview could not load. Move the heading to retry.';console.error(error)}finally{displayBusy=false}}
function setHeading(value,{fromPlay=false,row=null}={}){angle=Math.round(Math.max(0,Math.min(360,value)));$('heading').value=angle;$('heading-text').textContent=D.seriesMeta[color].dataset==='sim2'?`${angle}° / ?°`:`${angle}° / ${Math.round(requestedElevation)}°`;updateViewPad();state();requestDisplay(row||lookup[color][angle%360]);if(!fromPlay){continuous=angle;syncUrl()}}
function variance(){const ax=mode==='3d'?[0,1,2]:[+mode[0],+mode[1]];$('variance').textContent=`${(ax.reduce((s,i)=>s+D.ratio[i],0)*100).toFixed(1)}% of original-fit variation shown · fixed basis`}
function stop(){playing=false;lastFrame=0;$('play').textContent='▶ Play';$('play').setAttribute('aria-pressed','false');state();syncUrl()}
function animate(now){if(!playing)return;if(lastFrame)continuous+=(now-lastFrame)/1000*Number($('speed').value);lastFrame=now;if(continuous>360)continuous%=360;if(now-lastUpdate>=1000/30){lastUpdate=now;const next=Math.round(continuous);if(next!==angle)setHeading(next,{fromPlay:true})}requestAnimationFrame(animate)}
function pickColor(next,{addToPlot=true}={}){color=next;if(addToPlot){visibleSeries.add(color);syncSeriesCheckboxes()}document.documentElement.style.setProperty('--accent',D.colors[color]);$('series').value=color;$('elevation').disabled=D.seriesMeta[color].dataset==='sim2';$('elevation').value=D.seriesMeta[color].dataset==='sim2'?'unknown':String(elevation);drawViewSamples();setHeading(angle);scheduleStatic()}
function closest(x,y){let best=null,d=14*14;for(const p of screenPoints){const distance=(p.x-x)**2+(p.y-y)**2;if(distance<d){d=distance;best=p.row}}return best}
let pointer=null;
overlay.addEventListener('pointerdown',e=>{pointer={x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY,moved:false};overlay.setPointerCapture(e.pointerId)});
overlay.addEventListener('pointermove',e=>{const rect=overlay.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;if(pointer){const dx=e.clientX-pointer.x,dy=e.clientY-pointer.y;if(Math.abs(e.clientX-pointer.startX)+Math.abs(e.clientY-pointer.startY)>2)pointer.moved=true;pointer.x=e.clientX;pointer.y=e.clientY;if(mode==='3d'){yaw+=dx*.008;pitch=Math.max(-1.3,Math.min(1.3,pitch+dy*.008));scheduleStatic()}tooltip.style.display='none';return}const row=closest(x,y);tooltip.style.display=row?'block':'none';if(row){tooltip.textContent=`${row.category} · ${row.heading}° / ${row.elevation??'?'}°`;tooltip.style.left=Math.min(x+12,plotWidth-110)+'px';tooltip.style.top=y+12+'px'}});
overlay.addEventListener('pointerup',e=>{const moved=pointer?.moved;pointer=null;if(moved)return;const rect=overlay.getBoundingClientRect(),row=closest(e.clientX-rect.left,e.clientY-rect.top);if(row){stop();if(row.category!==color)pickColor(row.category);if(row.elevation!==null)setElevation(row.elevation);setHeading(row.heading,{row})}});
overlay.addEventListener('pointercancel',()=>pointer=null);overlay.addEventListener('pointerleave',()=>tooltip.style.display='none');overlay.addEventListener('wheel',e=>{if(mode!=='3d')return;e.preventDefault();zoom=Math.max(.5,Math.min(2,zoom*(e.deltaY>0?.9:1.1)));scheduleStatic()},{passive:false});

// The view pad selects recorded pixels/vectors, never interpolated embeddings.
const viewPad=$('view-pad'),svgNS='http://www.w3.org/2000/svg';
const padBounds={left:42,right:442,top:14,bottom:120},gradientImages=new Map();
function viewGradient(unknown){const key=palette+'/'+unknown;if(gradientImages.has(key))return gradientImages.get(key);const canvas=document.createElement('canvas');canvas.width=361;canvas.height=96;const brush=canvas.getContext('2d');for(let y=0;y<96;y++)for(let x=0;x<361;x++){brush.fillStyle=unknown&&(y<42||y>54)?'#172232':viewpointColor(x,unknown?null:D.elevations.at(-1)-y*(D.elevations.at(-1)-D.elevations[0])/95);brush.fillRect(x,y,1,1)}const image=canvas.toDataURL('image/png');gradientImages.set(key,image);return image}
const padX=value=>padBounds.left+(padBounds.right-padBounds.left)*value/360;
const padY=value=>padBounds.bottom-(padBounds.bottom-padBounds.top)*(value-D.elevations[0])/(D.elevations.at(-1)-D.elevations[0]);
function svgNode(name,attrs,text){const node=document.createElementNS(svgNS,name);for(const [key,value] of Object.entries(attrs))node.setAttribute(key,value);if(text!==undefined)node.textContent=text;return node}
function drawViewGrid(){const grid=$('view-grid');for(const h of [0,90,180,270,360]){const x=padX(h);grid.append(svgNode('line',{x1:x,y1:padBounds.top,x2:x,y2:padBounds.bottom,class:'grid'}),svgNode('text',{x,y:138,'text-anchor':h===360?'end':'middle'},h+'°'))}for(const e of D.elevations){const y=padY(e);grid.append(svgNode('line',{x1:padBounds.left,y1:y,x2:padBounds.right,y2:y,class:'grid'}),svgNode('text',{x:34,y:y+3,'text-anchor':'end'},e+'°'))}}
function drawViewSamples(){const unknown=D.seriesMeta[color].dataset==='sim2';$('view-color-anchors').dataset.unknown=String(unknown);$('view-gradient').setAttribute('href',viewGradient(unknown));const dots=$('view-samples');dots.replaceChildren();for(const row of fullGroups[color])dots.append(svgNode('circle',{cx:padX(row.heading),cy:row.elevation===null?67:padY(row.elevation),r:2,fill:viewpointColors.get(row.embedding_row),'data-row':row.embedding_row}))}
function updateViewPad(){if(!color)return;const unknown=D.seriesMeta[color].dataset==='sim2';$('view-request').setAttribute('transform',`translate(${padX(angle)} ${unknown?67:padY(requestedElevation)})`);const row=selected;if(row){$('view-recorded').style.fill=viewpointColors.get(row.embedding_row);$('view-recorded').setAttribute('cx',padX(row.heading));$('view-recorded').setAttribute('cy',row.elevation===null?67:padY(row.elevation));$('view-recorded').style.display=row.category===color?'':'none'}viewPad.setAttribute('aria-label',`Requested camera view: angle ${angle} degrees, elevation ${unknown?'uncalibrated':Math.round(requestedElevation)+' degrees'}`);$('view-pad-help').textContent=unknown?'Drag left/right for angle. Elevation is uncalibrated in Sim2; colors encode angle only. Arrow keys adjust angle.':'Drag left/right for angle and up/down for elevation. Crosshair: requested · outlined dot: recorded view. Arrow keys adjust the view; Shift uses larger steps.'}
function updateMismatch(row){const dh=circular(row.heading,angle),de=row.elevation===null?null:Math.abs(row.elevation-requestedElevation);$('mismatch').textContent=dh===0&&(de===0||de===null)?'Exact view':`Δ angle ${dh}°${de===null?'':` · elev ${de.toFixed(1)}°`}`}
function setElevation(value){if(D.seriesMeta[color].dataset==='sim2')return;requestedElevation=Math.max(D.elevations[0],Math.min(D.elevations.at(-1),value));const next=nearestElevation(requestedElevation);if(next!==elevation){elevation=next;rebuildViews();scheduleStatic()}$('elevation').value=String(elevation)}
let viewPointer=null,padFrame=0,padRequest=null;
function flushPad(){padFrame=0;if(!padRequest)return;const {heading,elevation:next}=padRequest;padRequest=null;setElevation(next);setHeading(heading)}
function requestPad(e){const rect=viewPad.getBoundingClientRect(),x=(e.clientX-rect.left)/rect.width*460,y=(e.clientY-rect.top)/rect.height*154;padRequest={heading:Math.max(0,Math.min(360,(x-padBounds.left)/(padBounds.right-padBounds.left)*360)),elevation:D.elevations[0]+(padBounds.bottom-y)/(padBounds.bottom-padBounds.top)*(D.elevations.at(-1)-D.elevations[0])};if(!padFrame)padFrame=requestAnimationFrame(flushPad)}
viewPad.addEventListener('pointerdown',e=>{if(e.button!==0)return;stop();viewPointer=e.pointerId;viewPad.setPointerCapture(e.pointerId);viewPad.focus({preventScroll:true});requestPad(e)});
viewPad.addEventListener('pointermove',e=>{if(e.pointerId===viewPointer)requestPad(e)});
viewPad.addEventListener('pointerup',e=>{if(e.pointerId!==viewPointer)return;requestPad(e);if(padFrame)cancelAnimationFrame(padFrame);flushPad();viewPointer=null});
viewPad.addEventListener('pointercancel',()=>{viewPointer=null;padRequest=null;if(padFrame)cancelAnimationFrame(padFrame);padFrame=0});
viewPad.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End'].includes(e.key))return;e.preventDefault();stop();const step=e.shiftKey?15:1;if(e.key==='ArrowUp'||e.key==='ArrowDown')setElevation(requestedElevation+(e.key==='ArrowUp'?step:-step));setHeading(e.key==='Home'?0:e.key==='End'?360:angle+(e.key==='ArrowLeft'?-step:e.key==='ArrowRight'?step:0))});
drawViewGrid();
function syncPaletteKey(){for(const name of anchorNames){const item=document.querySelector(`[data-view-anchor=${name}]`);item.value=rgbHex(viewAnchors[name]);item.title=item.value}}
function applyPalette(){
 viewLab=Object.fromEntries(Object.entries(viewAnchors).map(([name,rgb])=>[name,toOklab(rgb)]));
 viewpointColors=new Map(D.rows.map(row=>[row.embedding_row,viewpointColor(row.heading,row.elevation)]));
 gradientImages.clear();syncPaletteKey();drawViewSamples();updateViewPad();scheduleStatic();state();syncUrl();
}
for(const [name,entry] of Object.entries(viewPalettes))$('view-palette').add(new Option(entry.label,name));$('view-palette').value=palette;
$('view-palette').onchange=()=>{palette=$('view-palette').value;viewAnchors=viewPalettes[palette].anchors;applyPalette()};
let paletteFrame=0;
function editAnchorColor(e){
 const name=e.currentTarget.dataset.viewAnchor;
 if(!validHex(e.currentTarget.value))return;
 if(palette==='custom'&&rgbHex(customAnchors[name])===e.currentTarget.value.toLowerCase())return;
 if(palette!=='custom')for(const key of anchorNames)customAnchors[key]=[...viewAnchors[key]];
 customAnchors[name]=hexRgb(e.currentTarget.value);palette='custom';viewAnchors=customAnchors;$('view-palette').value=palette;
 // Editing viewpoint colors immediately applies them to the PCA points too.
 colorBy='viewpoint';$('point-color').value=colorBy;
 try{localStorage.setItem('vehicle-view-hsv-colors',JSON.stringify(Object.fromEntries(anchorNames.map(key=>[key,rgbHex(viewAnchors[key])]))))}catch{}
 if(!paletteFrame)paletteFrame=requestAnimationFrame(()=>{paletteFrame=0;applyPalette()});
}
for(const name of anchorNames){const input=document.querySelector(`[data-view-anchor=${name}]`);input.addEventListener('input',editAnchorColor);input.addEventListener('change',editAnchorColor)}
syncPaletteKey();

function syncSeriesCheckboxes(){for(const [category,entry] of seriesCheckboxes)entry.input.checked=visibleSeries.has(category)}
function setVisible(next){visibleSeries.clear();for(const c of next)visibleSeries.add(c);syncSeriesCheckboxes();scheduleStatic();state();syncUrl()}
const displayCategories=[...categories].sort((a,b)=>{const rank=c=>c.startsWith('Scans · jeep/')?0:c.startsWith('Scans · truck/')?1:c.startsWith('Scans · ')?2:3;return rank(a)-rank(b)||a.localeCompare(b)});
for(const category of displayCategories){
 $('series').add(new Option(category,category));const label=document.createElement('label'),input=document.createElement('input'),dot=document.createElement('span'),count=document.createElement('span');
 input.type='checkbox';input.setAttribute('aria-label','Show '+category);input.checked=visibleSeries.has(category);dot.className='dot';dot.style.background=D.colors[category];count.className='sample-count';count.textContent=fullGroups[category].length;
 label.append(input,dot,document.createTextNode(category.replace('Scans · ','').replace('Sim2 · Jeep / ','Sim2 Jeep / ')),count);$('legend').append(label);seriesCheckboxes.set(category,{input,label,count});
 input.onchange=()=>{if(input.checked)visibleSeries.add(category);else visibleSeries.delete(category);scheduleStatic();state();syncUrl()};
}
$('only-series').onclick=()=>setVisible([color]);$('all-series').onclick=()=>setVisible(categories);$('clear-series').onclick=()=>setVisible([]);
$('all-elevations').checked=allElevations;$('all-elevations').onchange=()=>{allElevations=$('all-elevations').checked;scheduleStatic();state();syncUrl()};
$('point-color').value=colorBy;$('point-color').onchange=()=>{colorBy=$('point-color').value;scheduleStatic();state();syncUrl()};
$('paths').checked=showPaths;$('paths').onchange=()=>{showPaths=$('paths').checked;scheduleStatic();state();syncUrl()};
for(const value of D.elevations)$('elevation').add(new Option(value+'°',String(value)));const unknownElevation=new Option('Uncalibrated · Sim2','unknown');unknownElevation.disabled=true;$('elevation').add(unknownElevation);$('elevation').value=elevation;$('series').onchange=e=>{stop();pickColor(e.target.value)};$('elevation').onchange=e=>{stop();setElevation(Number(e.target.value));setHeading(angle)};
$('heading').oninput=e=>{if(playing)stop();setHeading(Number(e.target.value))};for(const button of document.querySelectorAll('[data-angle]'))button.onclick=()=>{stop();setHeading(Number(button.dataset.angle))};
$('play').onclick=()=>{if(playing){stop();return}playing=true;continuous=angle;lastFrame=0;lastUpdate=0;$('play').textContent='Ⅱ Pause';$('play').setAttribute('aria-pressed','true');state();prefetch();requestAnimationFrame(animate)};
$('projection').value=mode;$('projection').onchange=()=>{stop();mode=$('projection').value;variance();state();staticPlot();syncUrl()};$('reset').onclick=()=>{yaw=-.65;pitch=.55;zoom=1;staticPlot()};
$('labels-link').href=D.labelsUrl;$('cache-link').href=D.cacheUrl;$('methods-link').href=D.methodsUrl;document.addEventListener('visibilitychange',()=>{if(document.hidden)stop()});
new ResizeObserver(resize).observe(plot);variance();pickColor(color,{addToPlot:false});resize();

const replay=$('camera-sample');for(const [i,row] of D.cameraSamples.entries())replay.add(new Option(`${row.observer} · t=${row.scene_time_sec}s · ${row.resampling.output_size_px.join('×')} px`,String(i)));
function showCameraSample(){const row=D.cameraSamples[Number(replay.value)];if(!row)return;$('camera-crop').src='camera/'+row.crop_path;$('camera-description').textContent=`Requested elevation ${row.request.elevation_deg.toFixed(1)}°, selected ${row.selected_elevation_deg}°; viewing-ray error ${row.viewing_ray_error_deg.toFixed(1)}°. Crop/native cosine ${row.cosine_to_native_view.toFixed(3)}. Actual pixel dimensions; enlarged preview uses nearest-neighbor display.`;window.cameraReplayState={observer:row.observer,time:row.scene_time_sec,crop:row.crop_path,size:row.resampling.output_size_px,error:row.viewing_ray_error_deg};selectedCameraSample=row;drawMarker();}replay.onchange=showCameraSample;showCameraSample();
