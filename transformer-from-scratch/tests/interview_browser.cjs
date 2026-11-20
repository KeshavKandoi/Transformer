const assert = require('node:assert/strict');
const path = require('node:path');
const {chromium} = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');

const base = 'http://127.0.0.1:' + (process.env.DEMO_PORT || '8765');
const output = process.env.DEMO_SCREENSHOTS;
const prompts = [
  ['   can  you help me   ', 'क्या तुम मेरी मदद कर सकते हो?'],
  ['MY NAME IS RAVI!', 'मेरा नाम रवि है।'],
  ['I have two books.', 'मेरे पास दो किताबें हैं।'],
];
async function run() {
  const browser = await chromium.launch({
    executablePath: process.env.CHROME_PATH || undefined,
    args: ['--no-sandbox'],
  });
  try {
    const context = await browser.newContext({
      viewport: {width: 1440, height: 900},
      permissions: ['clipboard-read', 'clipboard-write'],
    });
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', error=>errors.push(error.message));
    const screenshot = (name, locator=page) => output &&
      locator.screenshot({path:path.join(output,name+'.png'), animations:'disabled'});
    await page.goto(base);
    await page.locator('#gallery .example').first().waitFor();
    assert.equal(await page.title(), 'Transformer');
    assert.equal(await page.locator('h1').innerText(), 'Transformer');
    assert.equal(await page.locator('body').evaluate(body=>getComputedStyle(body).backgroundColor), 'rgb(255, 255, 255)');
    assert.equal(await page.locator('#gallery .example').count(),20);
    assert.equal(await page.locator('#gallery .miss').count(),9);
    assert.match(await page.locator('#stats').innerText(),/99\.17%[\s\S]*98\.72%[\s\S]*1,126,656[\s\S]*Epoch 11/);
    assert.match(await page.locator('#dimensions').innerText(),/2 encoder layers.*4 heads × 32/);
    const health=await (await context.request.get(base+'/api/health')).json();
    const info=await (await context.request.get(base+'/api/info')).json();
    assert.equal(health.status,'ready');
    assert.equal(info.checkpoint_sha256,health.checkpoint_sha256);
    const badRequest = await context.request.post(base+'/api/translate',{data:{text:9}});
    assert.equal(badRequest.status(),400);
    assert.match((await badRequest.json()).error,/English sentence/);
    const missing = await context.request.get(base+'/api/does-not-exist');
    assert.equal(missing.status(),404);
    for (const anchor of ['translate','architecture','attention','metrics','about']) {
      await page.locator(`nav a[href="#${anchor}"]`).click();
      assert.equal(new URL(page.url()).hash,'#'+anchor);
      assert.equal(await page.locator('#'+anchor).isVisible(),true);
    }
    for (const [source,hindi] of prompts) {
      await page.locator('#english').fill(source);
      const response=page.waitForResponse(r=>r.url().endsWith('/api/translate'));
      await page.locator('#english').press('Control+Enter');
      const result=await response;
      assert.equal(result.status(),200);
      assert.equal((await result.json()).outputs.beam.text,hindi);
      await page.locator('#prediction .prediction-text').waitFor();
      assert.equal(await page.locator('#prediction .prediction-text').innerText(),hindi);
    }
    await page.locator('#english').fill('Can you help me?');
    await page.locator('#strategy').selectOption('compare');
    await page.locator('#submit').click();
    await page.locator('#prediction .prediction-text').nth(1).waitFor();
    assert.deepEqual(await page.locator('#prediction .prediction-text').allInnerTexts(),
      ['क्या तुम मुझे फी मदद कर सकते हो?','क्या तुम मेरी मदद कर सकते हो?']);
    await page.locator('#strategy').selectOption('beam');
    await page.locator('#submit').click();
    await page.locator('#prediction .prediction-text').waitFor();
    await screenshot('translation',page.locator('.workbench'));
    await page.locator('#copy').click();
    assert.equal(await page.evaluate(()=>navigator.clipboard.readText()),prompts[0][1]);
    await screenshot('architecture',page.locator('#architecture'));
    await page.locator('#attentionType').selectOption('cross');
    await page.locator('#layer').selectOption('1');
    await page.locator('#head').selectOption('2');
    assert.ok(await page.locator('#matrix td').count()>0);
    await screenshot('attention',page.locator('#attention'));
    await page.locator('#attentionType').selectOption('causal_mask');
    assert.equal(await page.locator('#head').isDisabled(),true);
    assert.ok(await page.locator('#matrix td').filter({hasText:'·'}).count()>0);
    await page.locator('#progressMetric').selectOption('accuracy');
    assert.match(await page.locator('#progressStatus').innerText(),/Recorded/);
    await screenshot('metrics',page.locator('#metrics'));
    const card=page.locator('#gallery .example').first();
    const recorded=await card.locator('[lang=hi]').first().innerText();
    await card.click();
    await page.locator('#prediction .prediction-text').waitFor();
    assert.equal(await page.locator('#strategy').inputValue(),'greedy');
    assert.equal(await page.locator('#prediction .prediction-text').innerText(),recorded);

    for (const [input,message] of [
      ['',/Enter an English sentence/], ['a'.repeat(300),/outside/],
      ['Why is the sky blue?',/outside/], ['kaise ho',/Romanized Hindi/], ['नमस्ते',/Hindi input/],
    ]) {
      await page.locator('#english').fill(input);
      await page.locator('#submit').click();
      await page.locator('#status.warning').waitFor();
      assert.match(await page.locator('#status').innerText(),message);
      assert.equal(await page.locator('#prediction .prediction-text').count(),0);
    }
    await page.route('**/api/translate', async route=>{
      await new Promise(resolve=>setTimeout(resolve,400));
      await route.fulfill({status:500,body:'server error'});
    });
    await page.locator('#english').fill('Can you help me?');
    await page.locator('#submit').click();
    assert.equal(await page.locator('#submit').isDisabled(),true);
    assert.match(await page.locator('#status').innerText(),/Running/);
    await page.locator('#status.warning').waitFor();
    assert.match(await page.locator('#status').innerText(),/HTTP 500/);
    await page.unroute('**/api/translate');
    await page.locator('#clear').click();
    assert.equal(await page.locator('#english').inputValue(),'');

    delete info.aliases;
    info.metrics=null;
    info.gallery=[null,info.gallery[0]];
    info.curve=null;
    await page.route('**/api/info',route=>route.fulfill({json:info}));
    await page.reload();
    await page.locator('#gallery .example').first().waitFor();
    assert.equal(await page.locator('#gallery .example').count(),1);
    assert.match(await page.locator('#stats').innerText(),/Unavailable[\s\S]*98\.72%/);
    await page.unroute('**/api/info');
    for (const width of [320,390,768,1440]) {
      await page.setViewportSize({width,height:900});
      await page.goto(base);
      await page.locator('#gallery .example').first().waitFor();
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth),width);
    }
    await page.setViewportSize({width:320,height:900});
    await page.route('**/api/translate',route=>route.fulfill({json:{
      outputs:{beam:{text:'किताब'.repeat(100),eos:false,flags:{repetition:true}}},attention:null,
    }}));
    await page.locator('#english').fill('Can you help me?');
    await page.locator('#submit').click();
    await page.getByText('This output may be unreliable.').waitFor();
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth),320);
    assert.equal(await page.locator('#inspectControls').isHidden(),true);
    assert.deepEqual(errors,[]);
    await context.close();
    console.log('Browser smoke passed: live model, copy, masks, gallery, partial metadata, slow/error responses, and 4 viewport widths.');
  } finally {
    await browser.close();
  }
}
run().catch(error=>{console.error(error);process.exitCode=1;});
