'use strict';
const test=require('node:test');const assert=require('node:assert/strict');
const {send}=require('./upload.js');
function fakeXhr(status,body){return class {
  upload={addEventListener:(name,fn)=>{this.progress=fn;}};
  open(method,url){this.method=method;this.url=url;}
  setRequestHeader(){}
  send(file){this.progress({lengthComputable:true,loaded:file.size,total:file.size});this.status=status;this.responseText=JSON.stringify(body);this.onload();}
};}
test('reports upload bytes and accepts only a processing job ID',async()=>{
  const progress=[];const result=await send('/api/analyze',{size:12},value=>progress.push(value),fakeXhr(202,{status:'processing',job_id:'job-one'}));
  assert.equal(result.job_id,'job-one');assert.deepEqual(progress,[100]);
  await assert.rejects(send('/api/analyze',{size:12},null,fakeXhr(200,{status:'ok'})),/processing job ID/);
});
test('rejects empty input and HTTP errors without claiming a job',async()=>{
  await assert.rejects(send('/api/analyze',{size:0},null,fakeXhr(202,{status:'processing',job_id:'x'})),/non-empty/);
  await assert.rejects(send('/api/analyze',{size:12},null,fakeXhr(413,{error:'too large'})),/too large/);
});
