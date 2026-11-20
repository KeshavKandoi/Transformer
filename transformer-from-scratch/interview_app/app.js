'use strict';
const $ = (id) => document.getElementById(id);
let latest = null;
let curveRows = [];
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
  if (!isRecord(a) || !Array.isArray(a.source) || !Array.isArray(a.target_input) || !Array.isArray(a.target_emitted)) return false;
  const key = $('attentionType').value;
  const isMask = key.endsWith('_mask');
  const tensor = a[key];
  if (!Array.isArray(tensor) || !tensor.length) return false;
  let values, rows, columns;
  if (isMask) {
    const isPadding = key === 'padding_mask';
    values = isPadding ? [tensor] : tensor;
    rows = isPadding ? ['allowed'] : a.target_input;
    columns = key === 'causal_mask' ? a.target_input : a.source;
  } else {
    options($('layer'),tensor.length,'Layer');
    const layer = tensor[Number($('layer').value)];
    if (!Array.isArray(layer) || !layer.length) return false;
    options($('head'),layer.length,'Head');
    values = layer[Number($('head').value)];
    rows = key === 'encoder' ? a.source : a.target_input;
    columns = key === 'decoder' ? a.target_input : a.source;
  }
  if (!Array.isArray(values) || values.length !== rows.length ||
      !values.every(row => Array.isArray(row) && row.length === columns.length)) return false;
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
      cell.style.background = `rgba(66, 112, 211, ${Math.min(.48, number * .55)})`;
      cell.title = `${rows[i]} → ${columns[j]}: ${number.toFixed(4)}`;
      line.append(cell);
    });
    table.append(line);
  });
  $('matrix').replaceChildren(table);
  $('tokens').textContent = `Source: ${a.source.join(' · ')} | Emitted: ${a.target_emitted.join(' · ')} | Decode: ${latest.attention_for}`;
  return true;
}
function drawCurve(id, curve, series) {
  const canvas=$(id), context=canvas.getContext('2d');
  const w=canvas.width,h=canvas.height,m=76;
  context.clearRect(0,0,w,h);
  if (!curve.length) return;
  const high=Math.max(1e-8,...curve.flatMap(row=>series.map(([key])=>row[key])))*1.05;
  context.font='23px system-ui';
  context.strokeStyle='#d6dce7'; context.fillStyle='#3a3a3a';
  for(let i=0;i<=4;i++) {
    const y=h-42-i*(h-86)/4;
    context.beginPath();context.moveTo(m,y);context.lineTo(w-25,y);context.stroke();
    const value=high*i/4;
    context.fillText(value === 0 ? '0' : value < .01 ? value.toExponential(1) : value.toFixed(2),2,y+7);
  }
  series.forEach(([key,label,color], index)=>{
    context.strokeStyle=color;context.fillStyle=color;context.lineWidth=4;
    context.beginPath();
    curve.forEach((row,i)=>{
      const x=m+(w-m-25)*i/Math.max(1,curve.length-1),y=h-42-(h-86)*row[key]/high;
      if(i===0)context.moveTo(x,y);else context.lineTo(x,y);
    });
    context.stroke();context.fillText(label,m+index*220,29);
  });
  context.fillStyle='#3a3a3a';
  context.fillText(`Epoch ${curve[0].epoch}`,m,h-9);
  if (curve.length > 1) context.fillText(`Epoch ${curve.at(-1).epoch}`,w-120,h-9);
}
function drawProgress() {
  const key = $('progressMetric').value;
  const rows = curveRows.filter(row => Number.isFinite(row[key]));
  $('progressStatus').textContent = rows.length ? 'Recorded at each completed epoch.' : 'Series unavailable.';
  drawCurve('progress',rows,[[key,key === 'accuracy' ? 'Validation token accuracy' : 'Learning rate','#3456a4']]);
}
function resetOutput() {
  latest=null;
  $('copy').disabled=true;
  $('prediction').replaceChildren();
  $('matrix').replaceChildren(el('p','The heatmap will appear after translation.','hint'));
  $('tokens').textContent='';
  $('inspectControls').hidden=true;
}
async function translate(event) {
  event.preventDefault();
  const text=$('english').value.trim(), button=$('submit');
  if (button.disabled) return;
  resetOutput();
  if (!text) {
    $('status').className='warning';
    $('status').textContent='Enter an English sentence.';
    $('english').focus();
    return;
  }
  $('clear').disabled=true;
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
      const latency = Number.isFinite(out.milliseconds) ? ` · ${out.milliseconds} ms` : '';
      $('prediction').append(p,el('p',`${mode.toUpperCase()}${latency} · ${out.eos?'EOS':'length limit'}`,'minor'));
      if (out.flags?.repetition || out.flags?.replacement_character || out.flags?.control || out.flags?.empty) {
        $('prediction').append(el('p','This output may be unreliable.','minor warning'));
      }
    }
    const timings = Object.entries(latest.outputs).filter(([,out])=>Number.isFinite(out.milliseconds));
    if (timings.length) $('latency').textContent = 'Last CPU generation · ' +
      timings.map(([mode,out])=>`${mode} ${out.milliseconds} ms`).join(' · ');
    $('status').textContent=latest.source===latest.canonical_source?'':
      `Model input: ${latest.canonical_source}`;
    $('copy').disabled=!Object.values(latest.outputs).some(out=>out.text);
    if (matrix()) $('inspectControls').hidden=false;
  }catch(err){
    resetOutput();
    $('status').className='warning';$('status').textContent=err.message;
  }finally{button.disabled=false;$('clear').disabled=false;}
}
$('translateForm').addEventListener('submit',translate);
$('english').addEventListener('keydown',event=>{
  if (event.key==='Enter' && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    $('translateForm').requestSubmit();
  }
});
$('clear').addEventListener('click',()=>{
  $('english').value='';
  $('status').textContent='';
  resetOutput();
  $('english').focus();
});
$('copy').addEventListener('click',async()=>{
  try {
    await navigator.clipboard.writeText(Object.values(latest.outputs).map(out=>out.text).filter(Boolean).join('\n'));
    $('status').className='';
    $('status').textContent='Copied.';
  } catch {
    $('status').className='warning';
    $('status').textContent='Clipboard unavailable. Select the Hindi text to copy it.';
  }
});
for(const key of ['attentionType','layer','head'])$(key).addEventListener('change',matrix);
$('progressMetric').addEventListener('change',drawProgress);
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
    metricCard(Number.isInteger(info.epoch) ? `Epoch ${info.epoch}` : 'Unavailable', 'Best checkpoint'),
    metricCard(Number.isFinite(info.metrics?.chrf) ? info.metrics.chrf.toFixed(2) : 'Unavailable', 'Validation chrF2'),
    metricCard(Number.isFinite(info.metrics?.bleu) ? info.metrics.bleu.toFixed(2) : 'Unavailable', 'Validation BLEU'),
    metricCard(Number.isFinite(info.metrics?.eos) ? `${(100*info.metrics.eos).toFixed(2)}%` : 'Unavailable', 'Validation EOS completion')
  );
  $('checkpoint').textContent = typeof info.name === 'string' ? `Checkpoint · ${info.name}` : 'Checkpoint identity unavailable.';
  const config = info.architecture;
  const dimensions = ['num_encoder_layers','num_decoder_layers','num_heads','d_model','d_ff','max_seq_len'];
  $('dimensions').textContent = dimensions.every(key=>Number.isInteger(config?.[key])) ?
    `${config.num_encoder_layers} encoder layers · ${config.num_decoder_layers} decoder layers · ${config.num_heads} heads × ${config.d_model/config.num_heads} · model width ${config.d_model} · feed-forward width ${config.d_ff} · ${config.max_seq_len} positions` :
    'Checkpoint dimensions unavailable.';
  const slots = info.metrics?.slot_errors;
  const checks = isRecord(slots) ? Object.entries(slots)
    .filter(([,value])=>Number.isInteger(value?.errors) && Number.isInteger(value?.n))
    .map(([name,value])=>`${name} ${value.errors}/${value.n}`) : [];
  $('slotErrors').textContent = checks.length ? `Failed surface checks · ${checks.join(' · ')}` : 'Surface error checks unavailable.';
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
    const prediction = el('span',row.prediction || '(empty)');
    const reference = el('span',row.target);
    prediction.lang = reference.lang = 'hi';
    button.append(el('strong',row.source),el('span','Prediction','minor'),prediction,
      el('span','Reference','minor'),reference,
      el('span',`${row.exact?'Exact match':'Mismatch'} · ${row.family.replaceAll('-',' ')}`,row.exact?'match':'miss'),
      el('span','Synthetic validation','minor'));
    button.addEventListener('click',()=>{
      $('english').value=row.source;
      $('strategy').value='greedy';
      $('translateForm').requestSubmit();
      $('translateForm').scrollIntoView({block:'center'});
    });
    return button;
  }));
  if (!rows.length) $('gallery').append(el('p','Validation examples are unavailable.','minor'));
  curveRows = asList(info.curve).filter(row => isRecord(row) && Number.isFinite(row.epoch));
  const losses = curveRows.filter(row => Number.isFinite(row.train) && Number.isFinite(row.validation));
  drawCurve('curve',losses,[['train','Training','#3456a4'],['validation','Validation','#17664f']]);
  if (!losses.length) $('curveStatus').textContent = 'Training and validation loss data is unavailable.';
  drawProgress();
}).catch(err=>{
  $('stats').textContent=`Could not load checkpoint information. ${err.message}`;
  $('checkpoint').textContent='Checkpoint identity unavailable.';
  $('gallery').textContent='Validation examples are unavailable.';
  $('dimensions').textContent='Checkpoint dimensions unavailable.';
});
