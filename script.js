const worlds = [
  {name:'KEPLER-186F', short:'Kepler-186f', type:'TERRESTRIAL · CLASS M', distance:'557.7 LIGHT YEARS', coordinate:'06h 45m 09s   +23° 49′ 07″', eyebrow:'FIELD NOTE 001 — THE KEPLER FRONTIER', intro:'A field guide to the strange, beautiful worlds waiting just beyond what we know.', title:'A world in the habitable zone.', body:'Kepler-186f is an Earth-sized planet orbiting a cool red dwarf star. Its year lasts just 130 days. Somewhere beneath its quiet, copper sun, the first spring may be happening right now.', palette:'green'},
  {name:'HD 189733B', short:'HD 189733b', type:'HOT JUPITER · CLASS J', distance:'64.5 LIGHT YEARS', coordinate:'20h 00m 43s   +22° 42′ 39″', eyebrow:'FIELD NOTE 002 — THE BLUE TEMPEST', intro:'A cobalt world where the weather travels faster than sound, and rain falls sideways.', title:'It rains glass here. Sideways.', body:'HD 189733b is a deep-blue gas giant swept by winds reaching 8,700 kilometres per hour. Its brilliant color is no ocean: it comes from silicate particles in a scorching atmosphere, flung through the sky like glass in a hurricane.', palette:'blue'},
  {name:'55 CANCRI E', short:'55 Cancri e', type:'SUPER-EARTH · CLASS S', distance:'41 LIGHT YEARS', coordinate:'08h 52m 35s   +28° 19′ 51″', eyebrow:'FIELD NOTE 003 — THE DIAMOND SKY', intro:'A world so close to its star, its entire surface may be one vast, glowing ocean of lava.', title:'The ground is a kind of sky.', body:'Twice the size of Earth and locked in a blistering orbit, 55 Cancri e may be rich in carbon. Under immense pressure, some of that carbon could form diamond. Its dayside is a shimmering magma ocean, perpetually remade by starlight.', palette:'amber'},
  {name:'TRAPPIST-1E', short:'TRAPPIST-1e', type:'TERRESTRIAL · CLASS M', distance:'40 LIGHT YEARS', coordinate:'23h 06m 29s   −05° 02′ 29″', eyebrow:'FIELD NOTE 004 — THE RED HORIZON', intro:'A nearby, ancient world beneath a star that never sets. One of seven neighboring Earth-sized planets.', title:'Somewhere, the sun never moves.', body:'TRAPPIST-1e circles an ultra-cool red dwarf so closely that its year lasts just six Earth days. It may have liquid water. From its surface, neighboring worlds would hang enormous in the sky, each one a possible home for something unknown.', palette:'red'}
];
const $ = s => document.querySelector(s);
const planet = $('#planet');
let activeWorld = 0;
function selectWorld(index) {
  activeWorld = index;
  const w = worlds[index];
  document.body.dataset.palette = w.palette;
  $('#planetName').textContent = w.name;
  $('#planetType').textContent = w.type;
  $('#distance').textContent = w.distance;
  $('#coordinates').textContent = w.coordinate;
  $('#eyebrow').textContent = w.eyebrow;
  $('#intro').textContent = w.intro;
  document.querySelectorAll('.world-card').forEach((card, i) => card.classList.toggle('active', i === index));
  $('.strip-count span').textContent = `0${index + 1} / 04`;
  $('#modalTitle').textContent = w.title;
  $('#modalBody').textContent = w.body;
  $('#modalDistance').textContent = w.distance.replace('LIGHT ', '').replace(' YEARS',' YEARS');
  const palettes = {
    green: 'radial-gradient(ellipse at 30% 26%,#c0d2ac 0%,#66846d 17%,#315a5c 32%,#1e414e 51%,#142631 67%,#0b111d 80%)',
    blue: 'radial-gradient(ellipse at 30% 26%,#c7e5ef 0%,#519ac0 17%,#256c9c 34%,#253568 54%,#171d3a 72%,#090d19 85%)',
    amber: 'radial-gradient(ellipse at 30% 26%,#ffe2aa 0%,#e3a061 18%,#b85d46 35%,#673340 54%,#2c2035 72%,#100f1a 85%)',
    red: 'radial-gradient(ellipse at 30% 26%,#f2c27e 0%,#c87355 18%,#9c4947 35%,#553045 54%,#2b2037 72%,#100f1b 85%)'
  };
  planet.style.background = palettes[w.palette];
  const glows={green:'rgba(78,133,145,.17)',blue:'rgba(75,154,205,.25)',amber:'rgba(226,137,83,.22)',red:'rgba(207,91,77,.24)'};
  planet.style.boxShadow=`inset -48px -18px 78px 4px rgba(0,0,0,.88),inset 8px 10px 37px rgba(213,239,204,.24),0 0 80px ${glows[w.palette]}`;
}
document.querySelectorAll('.world-card').forEach(card => card.addEventListener('click', () => selectWorld(Number(card.dataset.world))));
$('#exploreButton').addEventListener('click', () => { $('#modal').classList.add('open'); $('#modal').setAttribute('aria-hidden','false'); });
function closeModal(){ $('#modal').classList.remove('open'); $('#modal').setAttribute('aria-hidden','true'); }
$('#closeModal').addEventListener('click', closeModal);
$('#modal').addEventListener('click', e => { if(e.target === $('#modal')) closeModal(); });
document.addEventListener('keydown', e => { if(e.key === 'Escape') closeModal(); });

// A quiet, synthesized low-frequency soundscape, created only after the listener asks for it.
let audioContext, droneNodes = [], soundOn = false;
$('#listenButton').addEventListener('click', async () => {
  const button = $('#listenButton');
  if (!soundOn) {
    audioContext ||= new (window.AudioContext || window.webkitAudioContext)();
    await audioContext.resume();
    const master = audioContext.createGain(); master.gain.setValueAtTime(0,audioContext.currentTime); master.gain.linearRampToValueAtTime(.055,audioContext.currentTime+2); master.connect(audioContext.destination);
    [55,55.7,82.4].forEach((frequency,i)=>{const osc=audioContext.createOscillator(),gain=audioContext.createGain();osc.type=i===2?'sine':'triangle';osc.frequency.value=frequency;gain.gain.value=i===2?.18:.4;osc.connect(gain);gain.connect(master);osc.start();droneNodes.push(osc,gain);});
    droneNodes.push(master); soundOn=true; button.querySelector('span:last-child').textContent='Mute the void'; button.classList.add('playing');
  } else {
    const master=droneNodes[droneNodes.length-1]; master.gain.cancelScheduledValues(audioContext.currentTime); master.gain.setTargetAtTime(0,audioContext.currentTime,.3);
    setTimeout(()=>{droneNodes.slice(0,-1).filter(n=>n.stop).forEach(n=>n.stop());droneNodes=[];},1200);
    soundOn=false; button.querySelector('span:last-child').textContent='Listen to the void'; button.classList.remove('playing');
  }
});
$('#menuButton').addEventListener('click', () => {
  let panel=$('.menu-panel');
  if(panel){panel.remove();return;}
  panel=document.createElement('nav');panel.className='menu-panel';panel.innerHTML='<a href="#top">THE FRONTIER</a><a href="#worldList">FIELD NOTES</a><a href="#worldList">TRANSMISSION LOG</a>';
  $('.topbar').append(panel); panel.querySelectorAll('a').forEach(a=>a.addEventListener('click',()=>panel.remove()));
});
const toast=(message)=>{const el=$('#toast');el.textContent=message;el.classList.add('show');setTimeout(()=>el.classList.remove('show'),2600)};
$('#year').textContent=new Date().getFullYear();
function updateClock(){ $('#clock').textContent=new Date().toISOString().slice(11,19)+' UTC'; }
updateClock();setInterval(updateClock,1000);

// Procedural starfield: tiny varied stars, slow parallax, and an occasional falling star.
const canvas=$('#stars'), ctx=canvas.getContext('2d');let width,height,stars=[],pointer={x:.5,y:.5};
function resize(){const dpr=Math.min(devicePixelRatio||1,2);width=innerWidth;height=innerHeight;canvas.width=width*dpr;canvas.height=height*dpr;canvas.style.width=width+'px';canvas.style.height=height+'px';ctx.setTransform(dpr,0,0,dpr,0,0);stars=Array.from({length:Math.floor(width*height/2300)},()=>({x:Math.random()*width,y:Math.random()*height,r:Math.random()*1.15+.15,a:Math.random()*.64+.13,p:Math.random()*Math.PI*2,s:Math.random()*.012+.002}));}
let shootingStar=null;
function draw(){ctx.clearRect(0,0,width,height);for(const s of stars){s.p+=s.s;const a=s.a*(.7+Math.sin(s.p)*.3);ctx.fillStyle=`rgba(222,231,239,${a})`;ctx.beginPath();ctx.arc(s.x+(pointer.x-.5)*s.r*11,s.y+(pointer.y-.5)*s.r*11,s.r,0,Math.PI*2);ctx.fill();}if(shootingStar){const m=shootingStar, travel=m.frame*5, headX=m.x+travel, headY=m.y+travel*.36;const gradient=ctx.createLinearGradient(headX-m.len*.4,headY-m.len*.145,headX,headY);gradient.addColorStop(0,'rgba(200,240,125,0)');gradient.addColorStop(1,'rgba(220,240,213,.92)');ctx.strokeStyle=gradient;ctx.lineWidth=1.2;ctx.beginPath();ctx.moveTo(headX,headY);ctx.lineTo(headX-m.len*.4,headY-m.len*.145);ctx.stroke();m.frame++;if(m.frame>28)shootingStar=null;}requestAnimationFrame(draw)}
addEventListener('resize',resize);addEventListener('pointermove',e=>{pointer.x=e.clientX/innerWidth;pointer.y=e.clientY/innerHeight;});resize();draw();
function meteor(){if(!shootingStar&&!document.hidden)shootingStar={x:Math.random()*width*.75+width*.1,y:Math.random()*height*.32,len:100+Math.random()*100,frame:0};setTimeout(meteor,6500+Math.random()*7500);}
setTimeout(meteor,4500);
