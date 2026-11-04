'use strict';
// Studio upload transport. XHR provides upload-byte progress; job progress comes from /api/job/{id}.
(function (root) {
  function send(url,file,onProgress,Xhr=root.XMLHttpRequest) {
    if(!file || !Number.isFinite(file.size) || file.size<=0) return Promise.reject(new TypeError('Choose a non-empty video file.'));
    return new Promise((resolve,reject)=>{
      const xhr=new Xhr();
      xhr.open('POST',url,true);
      xhr.timeout=120000;
      xhr.setRequestHeader('Content-Type','application/octet-stream');
      xhr.upload.addEventListener('progress',event=>{
        if(typeof onProgress==='function')onProgress(event.lengthComputable ? Math.min(100,Math.round(event.loaded/event.total*100)) : null);
      });
      xhr.onerror=()=>reject(new Error('Network error while uploading. Check the connection and retry.'));
      xhr.ontimeout=()=>reject(new Error('Upload timed out. Check Sessions before retrying to avoid a duplicate job.'));
      xhr.onabort=()=>reject(new Error('Upload cancelled.'));
      xhr.onload=()=>{
        let body;
        try{body=JSON.parse(xhr.responseText);}catch{reject(new Error(`Server returned HTTP ${xhr.status} without valid JSON.`));return;}
        if(xhr.status<200||xhr.status>=300||body?.error){reject(new Error(body?.error||`Upload failed with HTTP ${xhr.status}.`));return;}
        if(body?.status!=='processing'||typeof body.job_id!=='string'||!body.job_id){reject(new Error('Server did not return a processing job ID.'));return;}
        resolve(body);
      };
      xhr.send(file);
    });
  }
  const api=Object.freeze({send});
  root.TouchlineUpload=api;
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(typeof window!=='undefined'?window:globalThis);
