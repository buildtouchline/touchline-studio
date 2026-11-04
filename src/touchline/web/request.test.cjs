'use strict';
const test=require('node:test');const assert=require('node:assert/strict');
const {createGate,CancelledRead,TimedOutRead}=require('./request.js');
const deferred=()=>{let resolve,reject;const promise=new Promise((yes,no)=>{resolve=yes;reject=no;});return {promise,resolve,reject};};

test('a read that never settles times out and aborts its transport',async()=>{
  let signal;
  const gate=createGate((_,options)=>{signal=options.signal;return new Promise(()=>{});},x=>x,10);
  await assert.rejects(gate.read('/api/runs'),TimedOutRead);
  assert.equal(signal.aborted,true);
});
test('a newer read cancels the old read even if its transport ignores abort',async()=>{
  const first=deferred();const second=deferred();let calls=0;const signals=[];
  const gate=createGate((_,options)=>{signals.push(options.signal);return ++calls===1?first.promise:second.promise;},x=>x,1000);
  const old=gate.read('/api/run/old');await Promise.resolve();
  const current=gate.read('/api/run/current');await Promise.resolve();
  await assert.rejects(old,CancelledRead);assert.equal(signals[0].aborted,true);
  second.resolve({id:'current'});assert.deepEqual(await current,{id:'current'});
  first.resolve({id:'old'});
});
test('explicit cancellation and a stalled response parser both settle safely',async()=>{
  const gate=createGate(()=>Promise.resolve({}),()=>new Promise(()=>{}),10);
  await assert.rejects(gate.read('/api/runs'),TimedOutRead);
  const pending=gate.read('/api/runs');gate.cancel();
  await assert.rejects(pending,CancelledRead);
});
test('successful reads preserve the parsed response and can be repeated',async()=>{
  const gate=createGate(async url=>({url}),async response=>response.url,1000);
  assert.equal(await gate.read('/first'),'/first');
  assert.equal(await gate.read('/second'),'/second');
});
