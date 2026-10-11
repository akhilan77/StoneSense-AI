const { chromium } = require('c:/Users/akhil/projects/StoneSense-AI/frontend/node_modules/playwright');

async function testHospitalFed() {
  const browser = await chromium.launch({
    channel: 'chrome',
    headless: true,
  });

  const page = await browser.newPage({
    viewport: { width: 1920, height: 1080 }
  });

  const pageErrors = [];
  page.on('pageerror', err => pageErrors.push(err.message));

  console.log('Logging in as Hospital 1...');
  await page.goto('http://localhost:5173/login', { waitUntil: 'networkidle' });
  await page.click('button:has-text("Hospital 1")');
  await page.waitForURL('**/hospital-dashboard');

  console.log('Navigating to /hospital/federated-learning...');
  await page.click('a[href="/hospital/federated-learning"]');
  await page.waitForTimeout(3000);

  const title = await page.$eval('h1', el => el.textContent).catch(() => null);
  console.log('Page H1:', title);

  const textLength = await page.evaluate(() => document.body.innerText.length);
  console.log('Rendered body text length:', textLength);

  if (pageErrors.length > 0) {
    console.error('Page errors occurred:', pageErrors);
  } else {
    console.log('SUCCESS: Hospital federated learning loaded cleanly!');
  }

  await page.screenshot({ path: 'c:/Users/akhil/projects/StoneSense-AI/docs/report_figures/fig_9_6_hospital_federated_learning.png' });
  console.log('Updated fig_9_6_hospital_federated_learning.png with real rendered UI!');

  await browser.close();
}

testHospitalFed().catch(err => {
  console.error(err);
  process.exit(1);
});
