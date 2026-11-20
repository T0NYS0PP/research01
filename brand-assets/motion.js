/* Motion reconstructed from the archived Framer properties, independent of Framer. */
(() => {
  'use strict';
  const settings = JSON.parse(document.getElementById('motion-settings').textContent);
  const preference = matchMedia('(prefers-reduced-motion: reduce)');
  const finePointer = matchMedia('(hover:hover) and (pointer:fine)');
  let reduced = preference.matches;
  const clamp = (v, min=0, max=1) => Math.max(min, Math.min(max, v));
  const mix = (a,b,p) => a+(b-a)*p;
  const running = new Set();
  const animate = (element, frames, options={}) => {
    if (!element || reduced) return Promise.resolve();
    const animation = element.animate(frames, {duration:600,fill:'both',...options});
    running.add(animation);
    return animation.finished.catch(()=>{}).then(() => {
      running.delete(animation);
      // Final layout is owned by CSS/DOM; do not leave thousands of filled effects.
      animation.cancel();
    });
  };
  // Closed-form damped spring, using the mass/damping/stiffness from the source.
  function springFrames(from, {stiffness:k,damping:c,mass:m}, duration=1600, position=null) {
    const w=Math.sqrt(k/m), z=c/(2*Math.sqrt(k*m));
    return Array.from({length:101},(_,index)=>{
      const t=index/100*duration/1000;
      let displacement;
      if(z<1){const wd=w*Math.sqrt(1-z*z);displacement=Math.exp(-z*w*t)*(Math.cos(wd*t)+z*w/wd*Math.sin(wd*t));}
      else if(z===1)displacement=Math.exp(-w*t)*(1+w*t);
      else{const r1=-w*(z-Math.sqrt(z*z-1)),r2=-w*(z+Math.sqrt(z*z-1));displacement=(r2*Math.exp(r1*t)-r1*Math.exp(r2*t))/(r2-r1);}
      if(index===100)displacement=0;
      if(position)return {offset:index/100,transform:`translate3d(${position.to+(position.from-position.to)*displacement}%,0,0)`};
      return {offset:index/100,transform:`translate3d(0,${from*displacement}px,0)`,opacity:clamp(1-displacement,.001)};
    });
  }
  document.documentElement.classList.add('has-motion');

  // Source header: fixed orange navigation, dark translucent/blurred full-screen menu.
  const dialog=document.querySelector('#menu'), toggle=document.querySelector('.menu-toggle');
  let menuPhase='closed', menuToken=0;
  async function closeMenu() {
    if(!dialog.open || menuPhase==='closing')return;
    const token=++menuToken;menuPhase='closing';
    await animate(dialog,[{opacity:1,transform:'translateY(0)'},{opacity:0,transform:'translateY(-20px)'}],{duration:settings.menu.duration*.65,easing:settings.text.ease});
    if(token===menuToken){dialog.close();menuPhase='closed';toggle.focus();}
  }
  toggle.addEventListener('click',()=>{
    ++menuToken;menuPhase='open';dialog.showModal();toggle.setAttribute('aria-expanded','true');
    document.body.classList.add('menu-open');dialog.querySelector('.menu-close').focus();
    animate(dialog,[{opacity:0,transform:'translateY(-20px)'},{opacity:1,transform:'translateY(0)'}],{duration:settings.menu.duration,easing:settings.text.ease});
    dialog.querySelectorAll('.menu-content nav a,.menu-aside').forEach((el,i)=>animate(el,[{opacity:0,filter:`blur(${settings.menu.blur}px)`,transform:'translateY(18px)'},{opacity:1,filter:'blur(0px)',transform:'translateY(0)'}],{duration:600,delay:100+i*60,easing:settings.text.ease}));
  });
  dialog.querySelector('.menu-close').addEventListener('click',closeMenu);
  dialog.addEventListener('cancel',event=>{event.preventDefault();closeMenu();});
  // Restore focus when closing, not in the later close event: the user may have
  // already moved to a tab or another control before that event is dispatched.
  dialog.addEventListener('close',()=>{menuPhase='closed';toggle.setAttribute('aria-expanded','false');document.body.classList.remove('menu-open');});
  dialog.addEventListener('keydown',event=>{
    if(event.key!=='Tab')return;
    const items=[...dialog.querySelectorAll('a[href],button:not(:disabled),[tabindex="0"]')].filter(item=>item.getClientRects().length);
    if(event.shiftKey&&document.activeElement===items[0]){event.preventDefault();items.at(-1).focus();}
    else if(!event.shiftKey&&document.activeElement===items.at(-1)){event.preventDefault();items[0].focus();}
  });
  dialog.querySelectorAll('a').forEach(link=>link.addEventListener('click',()=>{dialog.close();}));
  const clock=document.querySelector('.menu-clock');
  function updateClock(){if(clock)clock.textContent=new Intl.DateTimeFormat(undefined,{hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}).format(new Date());}
  updateClock();setInterval(()=>{if(dialog.open&&!document.hidden)updateClock();},1000);

  // Crossfade and height transition between the business equivalents of source plans.
  const tabs=[...document.querySelectorAll('[role="tab"]')];
  let tabToken=0;
  function selectTab(tab) {
    if(tab.getAttribute('aria-selected')==='true')return;
    const token=++tabToken,container=document.querySelector('.service-panels'),oldHeight=container.getBoundingClientRect().height;
    tabs.forEach(item=>{const selected=item===tab;item.setAttribute('aria-selected',String(selected));item.tabIndex=selected?0:-1;document.getElementById(item.getAttribute('aria-controls')).hidden=!selected;});
    const panel=document.getElementById(tab.getAttribute('aria-controls')),newHeight=container.getBoundingClientRect().height;
    container.getAnimations().forEach(a=>a.cancel());
    animate(container,[{height:oldHeight+'px'},{height:newHeight+'px'}],{duration:450,easing:settings.text.ease});
    animate(panel,[{opacity:0,transform:'translateY(18px)'},{opacity:1,transform:'translateY(0)'}],{duration:600,easing:settings.text.ease});
    const photo=document.querySelector('.consult-photo img');
    if(photo){const src=tab.dataset.image;animate(photo,[{opacity:1},{opacity:0}],{duration:150}).then(()=>{if(token!==tabToken)return;photo.src=src;animate(photo,[{opacity:0,transform:'scale(1.3)'},{opacity:1,transform:'scale(1)'}],{duration:800,easing:settings.text.ease});});}
  }
  tabs.forEach((tab,index)=>{
    tab.addEventListener('click',()=>selectTab(tab));
    tab.addEventListener('keydown',event=>{
      let next;
      if(['ArrowRight','ArrowDown'].includes(event.key))next=(index+1)%tabs.length;
      if(['ArrowLeft','ArrowUp'].includes(event.key))next=(index-1+tabs.length)%tabs.length;
      if(event.key==='Home')next=0;if(event.key==='End')next=tabs.length-1;
      if(next===undefined)return;event.preventDefault();selectTab(tabs[next]);tabs[next].focus();
    });
  });

  // Spring accordion with an interruptible height transition; native semantics remain.
  document.querySelectorAll('.faq-list details').forEach(details=>{
    const summary=details.querySelector('summary');let animation=null,targetOpen=details.open;
    summary.addEventListener('click',event=>{
      if(reduced)return;event.preventDefault();
      if(!animation)targetOpen=details.open;
      const start=details.getBoundingClientRect().height;targetOpen=!targetOpen;
      animation?.cancel();details.style.height='';details.open=true;
      const finish=targetOpen?details.getBoundingClientRect().height:summary.getBoundingClientRect().height+1;
      details.style.overflow='hidden';details.classList.toggle('is-expanding',targetOpen);
      const frames=settings.faq.progress.map((p,index)=>({offset:index/(settings.faq.progress.length-1),height:mix(start,finish,p)+'px'}));
      const transition=details.animate(frames,{duration:settings.faq.duration,easing:'linear'});animation=transition;running.add(transition);
      const finishTransition=()=>{running.delete(transition);if(animation!==transition)return;details.open=targetOpen;details.style.overflow='';details.classList.remove('is-expanding');animation=null;};
      transition.onfinish=finishTransition;transition.oncancel=finishTransition;
    });
  });

  // Split visible text without changing its accessible name, line breaks or emphasis.
  function splitText(element) {
    const label=element.textContent;const walker=document.createTreeWalker(element,NodeFilter.SHOW_TEXT);const nodes=[];
    while(walker.nextNode())nodes.push(walker.currentNode);
    element.setAttribute('aria-label',label);
    nodes.forEach(node=>{
      const fragment=document.createDocumentFragment();
      // Keep English words together while allowing Japanese graphemes to wrap normally.
      const parts=node.textContent.match(/[A-Za-z0-9’'.,?!&—-]+|\s+|[^A-Za-z0-9\s]/gu)||[];
      parts.forEach(part=>{
        if(/^\s+$/.test(part)){fragment.append(document.createTextNode(part));return;}
        const word=document.createElement('span');word.className='motion-word';word.setAttribute('aria-hidden','true');
        Array.from(part).forEach(char=>{const span=document.createElement('span');span.className='motion-char';span.textContent=char;word.append(span);});fragment.append(word);
      });node.replaceWith(fragment);
    });
  }
  const textTargets=[...document.querySelectorAll('.section-heading h2,.intro-grid h2,.faq-layout h2,.page-heading h1,.detail-copy h2,.statement-en,.footer-cta,.service-orb-title,.showcase-title,.step-en,.footer-visual h2')];
  const revealTargets=[...document.querySelectorAll('.reveal,.detail-list,.about-details,.contact-layout form,.eyebrow')].filter(el=>!textTargets.includes(el));
  const revealObserver=new IntersectionObserver(entries=>entries.forEach(entry=>{
    if(!entry.isIntersecting)return;const el=entry.target;revealObserver.unobserve(el);el.dataset.revealed='true';
    if(reduced)return;
    if(el.classList.contains('motion-text')){
      const chars=[...el.querySelectorAll('.motion-char')];
      const flip=el.classList.contains('step-en');
      chars.forEach((char,index)=>animate(char,[{opacity:.001,transform:flip?'rotateY(60deg)':`translateY(${settings.text.y}px)`},{opacity:1,transform:flip?'rotateY(0deg)':'translateY(0)'}],{duration:settings.text.duration,delay:index*settings.text.stagger,easing:settings.text.ease}));
    }else animate(el,springFrames(settings.reveal.y,settings.reveal),{duration:1600,delay:settings.reveal.delay,easing:'linear'});
  }),{threshold:settings.reveal.threshold});
  textTargets.forEach(el=>{splitText(el);el.classList.add('motion-text');revealObserver.observe(el);});
  revealTargets.forEach(el=>revealObserver.observe(el));
  const counters=new IntersectionObserver(entries=>entries.forEach(entry=>{
    if(!entry.isIntersecting)return;counters.unobserve(entry.target);const number=entry.target,end=Number(number.dataset.count);
    if(reduced)return;const start=performance.now();
    function tick(now){const p=clamp((now-start)/1000);number.textContent=String(Math.round(end*p)).padStart(2,'0');if(p<1&&!reduced)requestAnimationFrame(tick);else number.textContent=String(end).padStart(2,'0');}
    requestAnimationFrame(tick);
  }),{threshold:.5});document.querySelectorAll('[data-count]').forEach(el=>counters.observe(el));

  // Entrance timing and circle motif use the source's authored delays.
  const hero=document.querySelector('.hero'),wordmark=document.querySelector('.hero-wordmark');
  if(wordmark)document.body.append(wordmark);
  if(hero&&!reduced){
    animate(hero.querySelector('h1'),springFrames(settings.hero.y,settings.hero),{duration:1800,delay:settings.hero.headlineDelay,easing:'linear'});
    animate(hero.querySelector('.hero-signature'),springFrames(settings.hero.y,settings.hero),{duration:1800,delay:settings.hero.signatureDelay,easing:'linear'});
    animate(hero.querySelector('.hero-copy p'),springFrames(settings.hero.y,settings.hero),{duration:1800,delay:settings.hero.subtitleDelay,easing:'linear'});
    const circle=hero.querySelector('.hero-copy p i');
    const motifFrames=settings.hero.motifProgress.map((p,index)=>({offset:index/(settings.hero.motifProgress.length-1),transform:`translate3d(0,${settings.hero.motifY*(1-p)}px,0)`,opacity:clamp(p,.001)}));
    animate(circle,motifFrames,{duration:settings.hero.motifDuration,delay:settings.hero.motifDelay,easing:'linear'});
    animate(wordmark,[{opacity:0},{opacity:1}],{duration:1000,delay:settings.hero.wordmarkDelay,easing:settings.text.ease});
    let first=true;try{first=!sessionStorage.getItem('pluspect-motion-entry');sessionStorage.setItem('pluspect-motion-entry','1');}catch{}
    if(first){const curtain=document.createElement('div');curtain.className='entry-curtain';curtain.setAttribute('aria-hidden','true');document.body.append(curtain);animate(curtain,[{transform:'translateY(0)'},{transform:'translateY(-101%)'}],{duration:settings.loader.duration,easing:settings.loader.ease}).then(()=>curtain.remove());}
  }

  // All scroll-driven effects share one requestAnimationFrame; no continuous scroll loop.
  const pairs=[...document.querySelectorAll('.photo-pair')],showcases=[...document.querySelectorAll('.service-showcase')],parallax=[...document.querySelectorAll('[data-parallax]')];
  const orb=document.querySelector('.service-orb-scene'),film=document.querySelector('.film-scene');
  let scheduled=false;
  function scrollFrame(){
    scheduled=false;const vh=innerHeight,vw=innerWidth,y=scrollY;
    if(hero){
      const h=hero.offsetHeight,p=clamp(y/h),logoP=reduced?1:clamp(y/(h*2));
      hero.style.setProperty('--hero-scale',reduced?1:Math.max(.001,1-p));hero.style.setProperty('--hero-opacity',reduced?1:1-p);
      hero.style.setProperty('--hero-copy-y',(reduced?0:-100*p)+'px');hero.style.setProperty('--hero-copy-opacity',reduced?1:clamp(1-p*3));
      if(wordmark){
        const image=wordmark.querySelector('.brand-logo-image');
        if(image){
          const ratio=Number(wordmark.dataset.logoRatio)||Number(image.getAttribute('height'))/Number(image.getAttribute('width'));
          const initial=vw-(vw<810?40:100),end=vw<810?140:170;
          wordmark.style.setProperty('--logo-width',mix(initial,end,logoP)+'px');
          wordmark.style.setProperty('--logo-top',mix(h-77-initial*ratio,(70-end*ratio)/2,logoP)+'px');
        }else{
          const initial=vw*(vw<810?.183:.1865),end=vw<810?22:28;
          wordmark.style.setProperty('--logo-size',mix(initial,end,logoP)+'px');
          wordmark.style.setProperty('--logo-top',mix(h-77-initial*.88,vw<810?26:24,logoP)+'px');
        }
        wordmark.classList.toggle('is-docked',logoP===1);
      }
    }
    if(film){const r=film.getBoundingClientRect(),p=clamp(-r.top/Math.max(1,r.height-vh));film.style.setProperty('--film-scale',reduced?1:mix(1,.6,p));film.style.setProperty('--film-copy-scale',reduced?1:mix(.7,3,p));}
    pairs.forEach((pair,index)=>{const r=pair.getBoundingClientRect(),p=clamp((vh-r.top)/(vh+r.height*.2)),travel=vw<810?settings.photoPair.phoneTravel:settings.photoPair.travel;pair.style.setProperty('--pair-offset',(reduced?0:(1-p)*travel*(index%2?-1:1))+'px');});
    showcases.forEach(el=>{const r=el.getBoundingClientRect(),p=clamp((vh-r.top)/(vh+r.height));el.style.setProperty('--showcase-y',(reduced?0:mix(-14,14,p))+'%');});
    parallax.forEach(el=>{const r=el.getBoundingClientRect(),p=clamp((vh-r.top)/(vh+r.height));el.style.setProperty('--parallax-y',(reduced?0:mix(-10,10,p))+'%');});
    if(orb){const r=orb.getBoundingClientRect(),p=clamp((vh*.6-r.top)/(r.height-vh*.4));orb.style.setProperty('--orb-scale',reduced?9:mix(0,settings.circle.endScale,p));orb.style.setProperty('--orb-text-scale',reduced?1:mix(.2,1,clamp(p*1.5)));orb.style.setProperty('--orb-text-opacity',reduced?1:clamp((p-.1)*3));}
  }
  function scheduleScroll(){if(!scheduled){scheduled=true;requestAnimationFrame(scrollFrame);}}
  addEventListener('scroll',scheduleScroll,{passive:true});addEventListener('resize',scheduleScroll,{passive:true});
  document.fonts.ready.then(scheduleScroll);scrollFrame();

  // Original 2.3-second process slideshow: stop outside viewport, on hover and focus.
  const carousel=document.querySelector('.process-carousel');
  if(carousel){
    const track=carousel.querySelector('.carousel-track'),slides=[...track.children],status=carousel.querySelector('.carousel-position'),pause=carousel.querySelector('.carousel-pause');
    let index=0,timer=null,visible=false,hover=false,focused=false,paused=false;
    function show(next){
      const start=new DOMMatrixReadOnly(getComputedStyle(track).transform).m41/track.offsetWidth*100;
      index=(next+slides.length)%slides.length;const end=-index*100;
      track.getAnimations().forEach(a=>a.cancel());track.style.transition='none';track.style.transform=`translate3d(${end}%,0,0)`;
      if(start!==end)animate(track,springFrames(0,settings.carousel,1500,{from:start,to:end}),{duration:1500,easing:'linear'});
      carousel.dataset.slide=String(index);status.textContent=String(index+1).padStart(2,'0')+' / '+String(slides.length).padStart(2,'0');slides.forEach((slide,i)=>slide.setAttribute('aria-hidden',String(i!==index)));
    }
    function sync(){clearInterval(timer);timer=null;if(visible&&!hover&&!focused&&!paused&&!preference.matches&&!document.hidden)timer=setInterval(()=>show(index+1),settings.carousel.interval);}
    new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;sync();},{threshold:.2}).observe(carousel);
    carousel.addEventListener('mouseenter',()=>{hover=true;sync();});carousel.addEventListener('mouseleave',()=>{hover=false;sync();});
    carousel.addEventListener('focusin',()=>{focused=true;sync();});carousel.addEventListener('focusout',event=>{if(!carousel.contains(event.relatedTarget)){focused=false;sync();}});
    pause.addEventListener('click',()=>{paused=!paused;pause.setAttribute('aria-pressed',String(paused));pause.textContent=paused?'再生':'一時停止';sync();});
    carousel.querySelector('.carousel-next').addEventListener('click',()=>{show(index+1);sync();});carousel.querySelector('.carousel-prev').addEventListener('click',()=>{show(index-1);sync();});
    document.addEventListener('visibilitychange',sync);preference.addEventListener('change',sync);show(0);
  }
  // Ticker loop uses duplicated, aria-hidden content; controls pause all automatic motion.
  document.querySelectorAll('.visual-ticker').forEach(ticker=>{
    const button=ticker.querySelector('.ticker-pause');button.addEventListener('click',()=>{const paused=ticker.classList.toggle('is-paused');button.setAttribute('aria-pressed',String(paused));button.textContent=paused?'再生':'一時停止';});
  });

  // Play muted only while the film is visible, including on phones. No player chrome.
  const reel=document.querySelector('#reel-dialog');
  const videos=[...document.querySelectorAll('video[data-source-film]')];
  const syncInlineFilms=[];
  videos.forEach(video=>{
    const phoneMedia=matchMedia('(max-width:809px)'),desktopSrc=video.getAttribute('src');
    let inView=false;
    function shouldPlay(){
      const rect=video.getBoundingClientRect();
      return inView&&rect.bottom>0&&rect.top<innerHeight&&!document.hidden&&!preference.matches&&!reel?.open&&!dialog.open;
    }
    function sync(){
      const visible=shouldPlay();
      video.autoplay=visible;
      if(!visible){video.pause();return;}
      video.play().then(()=>{if(!shouldPlay())video.pause();}).catch(()=>{
        // A browser may refuse autoplay; keep the poster and WATCH FILM action.
        // Never reveal native controls as an autoplay fallback.
      });
    }
    function configure(){
      const src=phoneMedia.matches&&video.dataset.phoneSrc?video.dataset.phoneSrc:desktopSrc;
      if(video.getAttribute('src')!==src){video.pause();video.src=src;}
      video.controls=false;video.muted=true;video.defaultMuted=true;video.loop=true;video.playsInline=true;
      sync();
    }
    configure();phoneMedia.addEventListener('change',configure);preference.addEventListener('change',configure);
    const observer=new IntersectionObserver(entries=>{
      inView=entries[0].isIntersecting&&entries[0].intersectionRatio>=.01;
      video.dataset.inView=String(inView);sync();
    },{threshold:.01});observer.observe(video);
    document.addEventListener('visibilitychange',sync);
    toggle.addEventListener('click',sync);dialog.addEventListener('close',sync);
    syncInlineFilms.push(sync);
  });
  if(reel){let opener;const video=reel.querySelector('video');
    video.controls=false;video.muted=true;video.defaultMuted=true;video.playsInline=true;video.loop=true;
    function syncReel(){if(reel.open&&!document.hidden)video.play().catch(()=>{});else video.pause();}
    document.querySelectorAll('[data-open-reel]').forEach(button=>button.addEventListener('click',()=>{opener=button;reel.showModal();document.body.classList.add('reel-open');video.currentTime=0;syncInlineFilms.forEach(sync=>sync());syncReel();}));
    reel.querySelector('.reel-close').addEventListener('click',()=>reel.close());
    reel.addEventListener('close',()=>{video.pause();document.body.classList.remove('reel-open');opener?.focus();syncInlineFilms.forEach(sync=>sync());});
    document.addEventListener('visibilitychange',syncReel);
  }

  // Source circular cursor is limited to image links and fine pointing devices.
  const cursor=document.createElement('div');cursor.className='motion-cursor';cursor.setAttribute('aria-hidden','true');document.body.append(cursor);
  document.addEventListener('pointermove',event=>{
    const target=event.target.closest('[data-cursor]');const active=target&&finePointer.matches&&!reduced&&!dialog.open;
    cursor.classList.toggle('is-active',!!active);if(!active)return;
    cursor.textContent=target.dataset.cursor;cursor.style.transform=`translate3d(${event.clientX}px,${event.clientY}px,0)`;
  },{passive:true});
  document.addEventListener('pointerleave',()=>cursor.classList.remove('is-active'));
  preference.addEventListener('change',()=>{
    reduced=preference.matches;document.documentElement.classList.toggle('motion-reduced',reduced);
    if(reduced){running.forEach(a=>a.cancel());document.querySelector('.entry-curtain')?.remove();videos.forEach(v=>v.pause());cursor.classList.remove('is-active');}
    scheduleScroll();
  });
  document.addEventListener('visibilitychange',()=>document.documentElement.classList.toggle('motion-paused',document.hidden));
})();
