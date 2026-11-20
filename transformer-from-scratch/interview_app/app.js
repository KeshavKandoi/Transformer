'use strict';
const $ = (id) => document.getElementById(id);
let latest = null;
const el = (tag, text, className) => {
  const item = document.createElement(tag);
  if (text !== undefined) item.textContent = text;
  if (className) item.className = className;
  return item;
};
const isRecord = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const asList = value => Array.isArray(value) ? value : [];
async function request(path, options) {
  const response = await fetch(path, options);
  const value = await response.json().catch(() => null);
  if (!response.ok) throw new Error(typeof value?.error === 'string' ? value.error : `Server request failed (HTTP ${response.status}).`);
  if (!isRecord(value)) throw new Error('The server returned an invalid response.');
  return value;
}
function options(select, count, name) {
  const previous = select.value;
  select.replaceChildren(...Array.from({length:count}, (_, i) => {
    const opt = el('option', `${name} ${i+1}`); opt.value = i; return opt;
  }));
  if (previous !== '' && Number(previous)<count) select.value=previous;
}
function matrix() {
  if (!latest) return;
  const a = latest.attention;
  const key = $('attentionType').value;
  const isMask = key.endsWith('_mask');
  const tensor = a[key];
  let values, rows, columns;
  if (isMask) {
    const isPadding = key === 'padding_mask';
    values = isPadding ? [tensor] : tensor;
    rows = isPadding ? ['allowed'] : a.target_input;
    columns = key === 'causal_mask' ? a.target_input : a.source;
  } else {
    options($('layer'),tensor.length,'Layer');
    options($('head'),tensor[Number($('layer').value)].length,'Head');
    values = tensor[Number($('layer').value)][Number($('head').value)];
    rows = key === 'encoder' ? a.source : a.target_input;
    columns = key === 'decoder' ? a.target_input : a.source;
  }
  $('layer').disabled = $('head').disabled = isMask;
  const table = el('table');
  const header = el('tr');
  header.append(el('th','Q / K'),...columns.map((t)=>el('th',t)));
  table.append(header);
  values.forEach((row,i) => {
    const line = el('tr');
    line.append(el('th',rows[i]));
    row.forEach((value,j) => {
      const number = Number(value);
      const cell = el('td',isMask ? (value?'✓':'·') : number.toFixed(2));
      cell.style.background = `rgba(113, 228, 192, ${Math.min(.68, number * .75)})`;
      cell.title = `${rows[i]} → ${columns[j]}: ${number.toFixed(4)}`;
      line.append(cell);
    });
    table.append(line);
  });
  $('matrix').replaceChildren(table);
  $('tokens').textContent = `Source: ${a.source.join(' · ')} | Emitted: ${a.target_emitted.join(' · ')} | Decode: ${latest.attention_for}`;
}
function drawCurve(curve) {
  if (!curve.length) return;
  const canvas=$('curve'), context=canvas.getContext('2d');
  const w=canvas.width,h=canvas.height,m=38;
  const high=Math.max(0.01,...curve.flatMap(r=>[r.train,r.validation]))*1.05;
  context.font='15px system-ui';
  context.strokeStyle='#36425d'; context.fillStyle='#a8b5ce';
  for(let i=0;i<=4;i++) {
    const y=h-m-i*(h-2*m)/4;
    context.beginPath();context.moveTo(m,y);context.lineTo(w-m,y);context.stroke();
    context.fillText((high*i/4).toFixed(2),2,y+4);
  }
  for(const [key,color,x] of [['train','#a3b6ff',55],['validation','#71e4c0',138]]) {
    context.strokeStyle=color;context.fillStyle=color;context.lineWidth=3;
    context.beginPath();
    curve.forEach((r,i)=>{
      const a=m+(w-2*m)*i/Math.max(1,curve.length-1),b=h-m-(h-2*m)*r[key]/high;
      if(i===0)context.moveTo(a,b);else context.lineTo(a,b);
    });
    context.stroke();context.fillText(key,x,20);
  }
  context.fillStyle='#a8b5ce';context.fillText('Epoch 1',m,h-10);
  context.fillText(`Epoch ${curve.at(-1).epoch}`,w-120,h-10);
}
async function translate(event) {
  event.preventDefault();
  const text=$('english').value, button=$('submit');
  button.disabled=true;$('status').className='';$('status').textContent='Running checkpoint…';
  try {
    latest=await request('/api/translate',{method:'POST',headers:{'Content-Type':'application/json'},
                                          body:JSON.stringify({text,strategy:$('strategy').value})});
    if (!isRecord(latest.outputs) || !Object.keys(latest.outputs).length ||
        !Object.values(latest.outputs).every(out => isRecord(out) && typeof out.text === 'string')) {
      throw new Error('The server returned no usable model prediction.');
    }
    $('prediction').replaceChildren();
    for(const [mode,out] of Object.entries(latest.outputs)){
      const p=el('p',out.text||'(empty prediction)','prediction-text');p.lang='hi';
      $('prediction').append(p,el('p',`${mode.toUpperCase()} · ${out.milliseconds} ms · ${out.eos?'EOS':'length limit'}`,'minor'));
    }
    $('status').textContent=latest.source===latest.canonical_source?'Recognized controlled pattern.':
      `Recognized controlled pattern. Model input: ${latest.canonical_source}`;
    $('inspectControls').hidden=false;matrix();
  }catch(err){
    latest=null;
    $('prediction').replaceChildren();
    $('matrix').replaceChildren();
    $('tokens').textContent='';
    $('inspectControls').hidden=true;
    $('status').className='warning';$('status').textContent=err.message;
  }finally{button.disabled=false;}
}
$('translateForm').addEventListener('submit',translate);
for(const key of ['attentionType','layer','head'])$(key).addEventListener('change',matrix);
function metricCard(value, label) {
  const item = el('article');
  item.append(el('strong', value), el('span', label));
  return item;
}
function exactMetric(metric, label, places) {
  const valid = Number.isFinite(metric?.exact) && metric.exact >= 0 && metric.exact <= 1 &&
                Number.isInteger(metric.n) && metric.n > 0;
  return metricCard(valid ? `${(metric.exact*100).toFixed(places)}%` : 'Unavailable',
                    valid ? `${label} · ${metric.n} rows` : label);
}
request('/api/info').then(info=>{
  $('stats').replaceChildren(
    exactMetric(info.metrics, 'Synthetic in-domain validation exact · greedy', 2),
    exactMetric(info.composition, 'Held-out composition exact · greedy', 2),
    metricCard(Number.isInteger(info.parameters) ? info.parameters.toLocaleString() : 'Unavailable', 'Trainable parameters'),
    metricCard(Number.isInteger(info.epoch) ? `Epoch ${info.epoch}` : 'Unavailable',
               typeof info.name === 'string' ? info.name : 'Checkpoint')
  );
  const families = asList(info.families).filter(name => typeof name === 'string');
  $('families').textContent = families.map(name=>name.replaceAll('-',' ')).join(' · ') ||
                             'Grammar coverage is unavailable for this checkpoint.';
  const aliases = isRecord(info.aliases) ? Object.entries(info.aliases)
    .filter(([,value]) => typeof value === 'string').map(([key,value])=>`${key} → ${value}`) : [];
  $('aliases').textContent=aliases.length?`Documented input aliases · ${aliases.join(' · ')}`:'';

  const rows = asList(info.gallery).filter(row => isRecord(row) &&
    ['source','prediction','target','family'].every(key => typeof row[key] === 'string') &&
    typeof row.exact === 'boolean');
  $('gallery').replaceChildren(...rows.map(row=>{
    const button=el('button',undefined,'example');button.type='button';
    button.append(el('strong',row.source),el('span',`Prediction · ${row.prediction||'(empty)'}`),
      el('span',`Generated reference · ${row.target}`),
      el('span',`${row.exact?'Exact match':'Mismatch'} · ${row.family.replaceAll('-',' ')}`,row.exact?'match':'miss'));
    button.addEventListener('click',()=>{
      $('english').value=row.source;
      $('strategy').value='greedy';
      $('translateForm').requestSubmit();
      $('translateForm').scrollIntoView({block:'center'});
    });
    return button;
  }));
  if (!rows.length) $('gallery').append(el('p','Validation examples are unavailable.','minor'));
  const curve = asList(info.curve).filter(row => isRecord(row) &&
    ['epoch','train','validation'].every(key => Number.isFinite(row[key])));
  if (curve.length) drawCurve(curve);
  else $('curveStatus').textContent = 'Training and validation loss data is unavailable.';
}).catch(err=>{
  $('stats').textContent=`Could not load checkpoint information. ${err.message}`;
  $('gallery').textContent='Validation examples are unavailable.';
});
fetch('/api/guide').then(r=>{
  if (!r.ok) throw new Error('Could not load the interview guide.');
  return r.text();
}).then(text=>{
  const guide=$('guideText');
  guide.replaceChildren();
  let list=null;
  for(const raw of text.split('\n')){
    const line=raw.trim();
    if(line.startsWith('# '))continue;
    if(line.startsWith('## ')){guide.append(el('h3',line.slice(3)));list=null;continue;}
    const numbered=line.match(/^\d+\. (.+)$/);
    const clean=(s)=>s.replaceAll('**','').replaceAll('`','');
    if(numbered){
      if(!list){list=el('ol');guide.append(list);}
      list.append(el('li',clean(numbered[1])));
    }else {
      list=null;
      if(line)guide.append(el('p',clean(line)));
    }
  }
}).catch(()=>{$('guideText').textContent='The interview guide is unavailable.';});
