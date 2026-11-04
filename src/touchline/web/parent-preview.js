'use strict';
const sampleToggle=document.getElementById('sampleToggle');
const sampleState=document.getElementById('sampleState');
sampleToggle.addEventListener('click',()=>{
  const visible=sampleState.hidden;
  sampleState.hidden=!visible;
  sampleToggle.setAttribute('aria-pressed',String(visible));
  sampleToggle.textContent=visible?'Hide illustrative layout':'Show illustrative layout';
  if(visible)sampleState.scrollIntoView({block:'nearest',behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});
});
const links=[...document.querySelectorAll('.parent-nav a')];
function activate(){const current=location.hash||'#overview';for(const link of links){const active=link.getAttribute('href')===current;link.classList.toggle('active',active);link.setAttribute('aria-current',active?'location':'false');}}
addEventListener('hashchange',activate);activate();
