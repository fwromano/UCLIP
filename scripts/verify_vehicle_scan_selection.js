async(page)=>{
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:8797/embedding.html');
 await page.waitForFunction(()=>window.orbitState?.selectedRow!==null&&window.orbitState?.pendingRow===null);
 await page.getByRole('button',{name:'Clear',exact:true}).click();
 await page.waitForFunction(()=>window.orbitPlotState?.rows===0&&window.orbitState?.selectedRow===null);
 const empty=await page.evaluate(()=>({row:document.querySelector('#jeep-preview').dataset.row,playDisabled:document.querySelector('#play').disabled,label:document.querySelector('#selected-view').textContent}));
 if(empty.row||!empty.playDisabled||empty.label!=='No vehicles on graph')throw new Error('Stale preview after clearing graph');
 const select=async category=>{
  await page.getByRole('checkbox',{name:'Show '+category,exact:true}).check();
  await page.waitForFunction(c=>window.orbitState?.previewSeries===c&&window.orbitState?.pendingRow===null,category);
  const match=await page.evaluate(()=>window.orbitState.markerRow===window.orbitState.imageRow&&window.orbitState.visibleSeries.includes(window.orbitState.previewSeries)&&document.querySelector('#jeep-preview').dataset.row===String(window.orbitState.imageRow));
  if(!match)throw new Error('Preview does not match graph selection');
 };
 await select('Scans · jeep/blue');
 await select('Scans · truck/red');
 await page.getByRole('checkbox',{name:'Show Scans · truck/red',exact:true}).uncheck();
 await page.waitForFunction(()=>window.orbitState?.previewSeries==='Scans · jeep/blue'&&window.orbitState?.pendingRow===null);
 await page.getByRole('checkbox',{name:'Show Scans · jeep/blue',exact:true}).uncheck();
 await page.waitForFunction(()=>window.orbitState?.selectedRow===null&&window.orbitPlotState?.rows===0);
 await page.locator('#series').selectOption('Scans · wolf/original');
 await page.waitForFunction(()=>window.orbitState?.previewSeries==='Scans · wolf/original'&&window.orbitState?.pendingRow===null&&window.orbitPlotState?.rows===168);
 await page.getByRole('button',{name:'Clear',exact:true}).click();
 await page.evaluate(()=>{
  for(const name of ['Scans · jeep/red','Scans · moose/original','Scans · jeep/green']){
   const checkbox=document.querySelector(`[aria-label="Show ${name}"]`);checkbox.checked=true;checkbox.dispatchEvent(new Event('change',{bubbles:true}));
  }
 });
 await page.waitForFunction(()=>window.orbitState?.previewSeries==='Scans · jeep/green'&&window.orbitState?.pendingRow===null);
 const final=await page.evaluate(()=>({preview:window.orbitState.previewSeries,visible:window.orbitState.visibleSeries,plotted:window.orbitPlotState.rows,aligned:window.orbitState.markerRow===window.orbitState.imageRow}));
 if(final.plotted!==504||!final.aligned)throw new Error(JSON.stringify(final));
 // Saved links with a hidden inspected series must preview a plotted vehicle.
 await page.goto('http://127.0.0.1:8797/embedding.html?color=Scans%20%C2%B7%20truck%2Fyellow&show=Scans%20%C2%B7%20jeep%2Fblue');
 await page.waitForFunction(()=>window.orbitState?.previewSeries==='Scans · jeep/blue'&&window.orbitState?.pendingRow===null);
 if(errors.length)throw new Error(JSON.stringify(errors));
 await page.evaluate(result=>window.graphPreviewVerification=result,{emptyClearsPreview:true,checkedVehicleSetsPreview:true,removedPreviewFallsBack:true,previewDropdownAddsToGraph:true,rapidSelectionKeepsLatest:true,savedLinkSelectsVisibleVehicle:true,final,errors});
}
