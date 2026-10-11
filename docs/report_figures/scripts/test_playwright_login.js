const path = require('path');
const { chromium } = require(path.resolve(__dirname, '../../../frontend/node_modules/playwright'));
const fs = require('fs');

async function testLogin() {
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  const context = await browser.newContext({ viewport: { width: 1920, height: 1080 } });
  const page = await context.newPage();

  console.log('Navigating to login page...');
  await page.goto('http://127.0.0.1:5173/login');
  await page.waitForLoadState('networkidle');

  console.log('Clicking Hospital 1 demo login button...');
  // The login page has demo account buttons with role 'Hospital 1'
  const hospBtn = page.locator('button', { hasText: 'Hospital 1' }).first();
  if (await hospBtn.count() > 0) {
    await hospBtn.click();
    console.log('Clicked Hospital 1 button');
  } else {
    console.log('Filling form manually...');
    await page.fill('input[type="email"]', 'hospital1@stonesense.ai');
    await page.fill('input[type="password"]', 'Hospital1!2026');
    await page.click('button[type="submit"]');
  }

  await page.waitForTimeout(2000);
  console.log('Current URL after login:', page.url());

  const token = await page.evaluate(() => localStorage.getItem('stonesense_access_token'));
  console.log('Stored token present:', !!token);

  await browser.close();
}

testLogin().catch(err => console.error('Error:', err));
