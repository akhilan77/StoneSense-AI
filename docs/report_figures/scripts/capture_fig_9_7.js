const { chromium } = require('c:/Users/akhil/projects/StoneSense-AI/frontend/node_modules/playwright');
const path = require('path');

const OUT_DIR = 'c:/Users/akhil/projects/StoneSense-AI/docs/report_figures';

async function main() {
  const browser = await chromium.launch({
    channel: 'chrome',
    headless: true,
  });

  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 },
    deviceScaleFactor: 1,
  });

  const page = await context.newPage();

  page.on('console', msg => console.log('BROWSER LOG:', msg.text()));
  page.on('pageerror', err => console.log('PAGE ERROR:', err.message));

  console.log('1. Navigating to login...');
  await page.goto('http://127.0.0.1:5173/login', { waitUntil: 'networkidle' });
  await page.waitForSelector('button:has-text("Developer")');

  console.log('2. Clicking Developer login...');
  await page.click('button:has-text("Developer")');
  await page.waitForURL('**/developer-dashboard');
  await page.waitForTimeout(3000);

  console.log('Current URL:', page.url());
  const bodySnippet = await page.evaluate(() => document.body.innerText.slice(0, 300));
  console.log('Developer Dashboard Body snippet:', bodySnippet);

  await page.screenshot({
    path: path.join(OUT_DIR, 'fig_9_7_developer_dashboard.png'),
    fullPage: false,
  });
  console.log('Saved fig_9_7_developer_dashboard.png successfully');

  await browser.close();
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
