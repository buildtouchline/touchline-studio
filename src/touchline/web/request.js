'use strict';
// One gate per read surface: a newer request cancels the previous read.
(function (root) {
  class CancelledRead extends Error {
    constructor() { super('Request replaced'); this.name='CancelledRead'; }
  }
  class TimedOutRead extends Error {
    constructor() { super('Request timed out. Retry the session.'); this.name='TimedOutRead'; }
  }
  function createGate(fetchImpl, parseResponse, timeoutMs=12000) {
    if (typeof fetchImpl!=='function' || typeof parseResponse!=='function' || !Number.isFinite(timeoutMs) || timeoutMs<=0) throw new TypeError('A fetch function, response parser and positive timeout are required');
    let active=null;
    function cancel() {
      if (!active) return;
      const request=active;active=null;
      request.reject(new CancelledRead());
      request.controller.abort();
    }
    async function read(url) {
      cancel();
      const controller=new AbortController();
      let rejectCancelled;
      const cancelled=new Promise((_,reject)=>{rejectCancelled=reject;});
      const request={controller,reject:rejectCancelled};active=request;
      let timer;
      const timedOut=new Promise((_,reject)=>{
        timer=setTimeout(()=>{reject(new TimedOutRead());controller.abort();},timeoutMs);
      });
      try {
        const response=Promise.resolve().then(()=>fetchImpl(url,{signal:controller.signal})).then(parseResponse);
        return await Promise.race([response,cancelled,timedOut]);
      } finally {
        clearTimeout(timer);
        if (active===request) active=null;
      }
    }
    return Object.freeze({read,cancel});
  }
  const api=Object.freeze({createGate,CancelledRead,TimedOutRead});
  root.TouchlineRequest=api;
  if (typeof module!=='undefined' && module.exports) module.exports=api;
})(typeof window!=='undefined' ? window : globalThis);
