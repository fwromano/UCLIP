async (page) => {
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:8797/embedding.html');
  await page.waitForFunction(()=>window.orbitState?.pendingRow===null&&window.orbitState?.selectedRow!==null);
  await page.locator('#point-color').selectOption('series');
  const choices={topLeft:'#ff0000',topRight:'#0000ff',bottomLeft:'#ffff00',bottomRight:'#00ff00'};
  await page.evaluate(values=>{
    for(const [name,value] of Object.entries(values)){
      const input=document.querySelector(`[data-view-anchor=${name}]`);
      input.value=value;input.dispatchEvent(new Event('change',{bubbles:true}));
    }
  },choices);
  await page.waitForFunction(()=>window.orbitState?.palette==='custom'&&window.orbitState?.interpolation==='OKLab');
  const result=await page.evaluate(async()=>{
    const parse=color=>color.match(/\d+/g).map(Number);
    let maximumAdjacentDifference=0;
    for(let y=0;y<=95;y++)for(let x=0;x<=180;x++){
      const a=parse(viewpointColor(x,y-5));
      if(x>0)maximumAdjacentDifference=Math.max(maximumAdjacentDifference,Math.hypot(...a.map((v,i)=>v-parse(viewpointColor(x-1,y-5))[i])));
      if(y>0)maximumAdjacentDifference=Math.max(maximumAdjacentDifference,Math.hypot(...a.map((v,i)=>v-parse(viewpointColor(x,y-6))[i])));
    }
    const image=new Image();image.src=document.querySelector('#view-gradient').getAttribute('href');await image.decode();
    const canvas=document.createElement('canvas');canvas.width=image.width;canvas.height=image.height;
    const context=canvas.getContext('2d');context.drawImage(image,0,0);
    const pixel=[...context.getImageData(121,47,1,1).data].slice(0,3);
    const expected=parse(viewpointColor(121,43));
    const row=JSON.parse(document.querySelector('#orbit-data').textContent).rows.find(r=>r.embedding_row===window.orbitState.selectedRow);
    const sample=document.querySelector(`#view-samples [data-row="${row.embedding_row}"]`);
    return {
      interpolation:window.orbitState.interpolation,anchors:window.orbitState.anchors,
      corners:{topLeft:parse(viewpointColor(0,90)),topRight:parse(viewpointColor(180,90)),bottomLeft:parse(viewpointColor(0,-5)),bottomRight:parse(viewpointColor(180,-5))},
      maximumAdjacentDifference,gradientPixel:pixel,expectedPixel:expected,
      gridMatches:pixel.every((v,i)=>v===expected[i]),
      sampleMatches:sample.getAttribute('fill')===viewpointColor(row.heading,row.elevation),
      markerMatches:window.orbitPoint.fill===viewpointColor(row.heading,row.elevation),
      selectedColorMatches:window.orbitState.selectedColor===viewpointColor(row.heading,row.elevation),
      colorBy:window.orbitState.colorBy,plotted:window.orbitPlotState.rows,
      wraparoundMatches:viewpointColor(0,43)===viewpointColor(360,43),
      imageMarkerAligned:window.orbitState.markerRow===window.orbitState.imageRow
    };
  });
  const expected={topLeft:[255,0,0],topRight:[0,0,255],bottomLeft:[255,255,0],bottomRight:[0,255,0]};
  if(JSON.stringify(result.corners)!==JSON.stringify(expected)||result.maximumAdjacentDifference>=40||!result.gridMatches||!result.sampleMatches||!result.markerMatches||!result.selectedColorMatches||result.colorBy!=='viewpoint'||result.plotted!==5268||!result.wraparoundMatches||!result.imageMarkerAligned||errors.length)throw new Error(JSON.stringify({result,errors}));
  await page.waitForTimeout(180);
  const shareUrl=page.url();await page.reload();
  await page.waitForFunction(()=>window.orbitState?.pendingRow===null&&window.orbitState?.selectedRow!==null);
  const restored=await page.evaluate(()=>window.orbitState.anchors);
  if(JSON.stringify(restored)!==JSON.stringify(result.anchors))throw new Error('Custom colors changed after reload');
  await page.evaluate(value=>window.sphereBlendVerification=value,{...result,urlRoundTrip:true,errors});
}
