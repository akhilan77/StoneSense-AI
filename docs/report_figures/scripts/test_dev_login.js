const { chromium } = require('c:/Users/akhil/projects/StoneSense-AI/frontend/node_modules/playwright');

async function testDevLogin() {
  const browser = await chromium.launch({
    channel: 'chrome',
    headless: true,
  });

  const page = await browser.newPage({
    viewport: { width: 1920, height: 1080 }
  });

  const pageErrors = [];
  page.on('pageerror', err => pageErrors.push(err.message));
  page.on('console', msg => {
    if (msg.type() === 'error') {
      console.log('BROWSER ERROR:', msg.text());
    }
  });

  console.log('Navigating to login page at http://localhost:5173/login...');
  await page.goto('http://localhost:5173/login', { waitUntil: 'networkidle' });

  console.log('Clicking Developer quick-login button...');
  await page.click('button:has-text("Developer")');

  console.log('Waiting for URL to change to /developer-dashboard...');
  await page.waitForURL('**/developer-dashboard', { timeout: 10000 });

  await page.waitForTimeout(3000);

  const title = await page.$eval('h1', el => el.textContent).catch(() => null);
  console.log('Page H1:', title);

  const textLength = await page.evaluate(() => document.body.innerText.length);
  console.log('Rendered body text length:', textLength);

  if (pageErrors.length > 0) {
    console.error('Page errors occurred:', pageErrors);
  } else {
    console.log('SUCCESS: No page errors! Developer dashboard loaded cleanly.');
  }

  // Take screenshot of working developer dashboard
  await page.screenshot({ path: 'c:/Users/akhil/projects/StoneSense-AI/docs/report_figures/fig_9_7_developer_dashboard.png' });
  console.log('Updated fig_9_7_developer_dashboard.png with real rendered UI!');

  await browser.close();
}

testDevLogin().catch(err => {
  console.error('Test failed:', err);
  process.exit(1);
});
