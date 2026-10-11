const path = require('path');
const { chromium } = require(path.resolve(__dirname, '../../../frontend/node_modules/playwright'));

const OUTPUT_DIR = path.resolve(__dirname, '..');
const BASE_URL = 'http://127.0.0.1:5173';

async function captureAllUIFigures() {
  console.log('Launching Chrome via Playwright...');
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 },
    deviceScaleFactor: 1
  });
  const page = await context.newPage();

  // 1. Figure 9.2: Landing + Login Page
  console.log('\n--- Capturing Figure 9.2: Landing / Login Page ---');
  await page.goto(`${BASE_URL}/login`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(1000);
  const fig9_2_path = path.join(OUTPUT_DIR, 'fig_9_2_landing_login.png');
  await page.screenshot({ path: fig9_2_path, fullPage: false });
  console.log(`Saved ${fig9_2_path}`);

  // 2. Log in as Hospital 1
  console.log('\n--- Logging in as Hospital 1 User ---');
  const hospBtn = page.locator('button', { hasText: 'Hospital 1' }).first();
  if (await hospBtn.count() > 0) {
    await hospBtn.click();
  } else {
    await page.fill('input[type="email"]', 'hospital1@stonesense.ai');
    await page.fill('input[type="password"]', 'Hospital1!2026');
    await page.click('button[type="submit"]');
  }
  await page.waitForTimeout(2000);

  // 3. Figure 9.3: Hospital Dashboard with Real Seeded Data
  console.log('\n--- Capturing Figure 9.3: Hospital Dashboard ---');
  await page.goto(`${BASE_URL}/hospital-dashboard`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(2000);
  const fig9_3_path = path.join(OUTPUT_DIR, 'fig_9_3_hospital_dashboard.png');
  await page.screenshot({ path: fig9_3_path, fullPage: false });
  console.log(`Saved ${fig9_3_path}`);

  // 4. Figure 9.4: Risk Prediction Page with SHAP Waterfall
  console.log('\n--- Capturing Figure 9.4: Risk Prediction with SHAP ---');
  await page.goto(`${BASE_URL}/risk-prediction/1`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(1000);

  // Fill in urine biomarker values from row 0 of kidneyData.csv
  console.log('Filling in clinical biomarker values from kidneyData.csv row 0...');
  // specific gravity = 1.021, ph = 4.91, osmo = 725, cond = 23.4, urea = 285, calc = 2.86
  try {
    const inputs = page.locator('input[type="number"]');
    const inputCount = await inputs.count();
    console.log(`Found ${inputCount} number inputs`);
    
    // Fill specific inputs by label text
    const fillByLabel = async (labelText, val) => {
      const label = page.locator('label', { hasText: labelText });
      if (await label.count() > 0) {
        const inp = label.locator('input');
        if (await inp.count() > 0) {
          await inp.fill(String(val));
        }
      }
    };

    await fillByLabel('Urine Specific Gravity', 1.021);
    await fillByLabel('Urine pH', 4.91);
    await fillByLabel('Osmolality', 725);
    await fillByLabel('Conductivity', 23.4);
    await fillByLabel('Urea', 285);
    await fillByLabel('Calcium', 2.86);

    // Click "Run assessment"
    const submitBtn = page.locator('button[type="submit"]', { hasText: 'assessment' }).first();
    if (await submitBtn.count() > 0) {
      console.log('Submitting risk assessment request...');
      await submitBtn.click();
      await page.waitForTimeout(3000);
    }
  } catch (e) {
    console.error('Error submitting risk assessment:', e);
  }

  const fig9_4_path = path.join(OUTPUT_DIR, 'fig_9_4_risk_prediction_shap.png');
  await page.screenshot({ path: fig9_4_path, fullPage: true });
  console.log(`Saved ${fig9_4_path}`);

  // 5. Figure 9.5: Stone Detection Page with Grad-CAM
  console.log('\n--- Capturing Figure 9.5: Stone Detection with Grad-CAM ---');
  await page.goto(`${BASE_URL}/stone-detection/1`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(1000);

  // Upload real Stone slice from test set
  const realStonePath = path.resolve(__dirname, '../../../dl/processed_grouped/test/Stone/Stone- (1004).jpg');
  console.log(`Setting file input: ${realStonePath}`);
  const fileInput = page.locator('input[type="file"]');
  if (await fileInput.count() > 0) {
    await fileInput.setInputFiles(realStonePath);
    await page.waitForTimeout(1500);

    // Click Run assessment
    const runBtn = page.locator('button[type="submit"]').first();
    if (await runBtn.count() > 0) {
      console.log('Submitting CT classification request...');
      await runBtn.click();
      await page.waitForTimeout(4000);
    }
  }

  const fig9_5_path = path.join(OUTPUT_DIR, 'fig_9_5_stone_detection_gradcam.png');
  await page.screenshot({ path: fig9_5_path, fullPage: true });
  console.log(`Saved ${fig9_5_path}`);

  // 6. Figure 9.6: Hospital Federated Learning Console
  console.log('\n--- Capturing Figure 9.6: Hospital Federated Learning ---');
  await page.goto(`${BASE_URL}/hospital/federated-learning`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(2000);
  const fig9_6_path = path.join(OUTPUT_DIR, 'fig_9_6_hospital_federated_learning.png');
  await page.screenshot({ path: fig9_6_path, fullPage: false });
  console.log(`Saved ${fig9_6_path}`);

  // 7. Figure 9.7: Developer Dashboard
  console.log('\n--- Capturing Figure 9.7: Developer Dashboard ---');
  // Log out and log in as Developer
  await page.goto(`${BASE_URL}/login`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(500);
  const devBtn = page.locator('button', { hasText: 'Developer' }).first();
  if (await devBtn.count() > 0) {
    await devBtn.click();
  } else {
    await page.fill('input[type="email"]', 'dev@stonesense.ai');
    await page.fill('input[type="password"]', 'StoneSenseDev!2026');
    await page.click('button[type="submit"]');
  }
  await page.waitForTimeout(2000);

  await page.goto(`${BASE_URL}/developer-dashboard`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(2500);
  const fig9_7_path = path.join(OUTPUT_DIR, 'fig_9_7_developer_dashboard.png');
  await page.screenshot({ path: fig9_7_path, fullPage: false });
  console.log(`Saved ${fig9_7_path}`);

  await browser.close();
  console.log('\nAll UI Screenshots successfully captured!');
}

captureAllUIFigures().catch(err => {
  console.error('Fatal error during capture:', err);
  process.exit(1);
});
