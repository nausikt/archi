import { test, expect } from '@playwright/test';
import { setupBasicMocks } from './fixtures';

// Follow tests/ui/chat.spec.ts for baseURL / beforeEach setup conventions.

test.describe('Playbooks', () => {
  test.beforeEach(async ({ page }) => {
    // Mock the playbooks API so tests are self-contained
    await setupBasicMocks(page);

    await page.route('**/api/playbooks*', async (route) => {
      const method = route.request().method();
      if (method === 'GET') {
        await route.fulfill({
          status: 200,
          json: { playbooks: [] },
        });
      } else {
        await route.fulfill({
          status: 200,
          json: { id: 1, name: 'pw-test-playbook', description: 'a test playbook', body: 'do the test thing' },
        });
      }
    });

    // Also mock agent list used by setActiveAgent
    await page.route('**/api/agents*', async (route) => {
      await route.fulfill({ status: 200, json: { agents: [] } });
    });

    await page.goto('/chat');
  });

  test('create a playbook via the panel and see it listed', async ({ page }) => {
    // Track whether the POST was made so we can mock a populated list on reload
    let playbookCreated = false;

    await page.route('**/api/playbooks*', async (route) => {
      const method = route.request().method();
      if (method === 'GET') {
        await route.fulfill({
          status: 200,
          json: playbookCreated
            ? { playbooks: [{ id: 1, name: 'pw-test-playbook', description: 'a test playbook', body: 'do the test thing' }] }
            : { playbooks: [] },
        });
      } else if (method === 'POST') {
        playbookCreated = true;
        await route.fulfill({
          status: 200,
          json: { id: 1, name: 'pw-test-playbook', description: 'a test playbook', body: 'do the test thing' },
        });
      } else {
        await route.fulfill({ status: 200, json: {} });
      }
    });

    // Open the playbooks modal directly (no running app — force display via JS)
    await page.locator('.playbooks-modal').evaluate((el: HTMLElement) => (el.style.display = 'flex'));

    // The editor is hidden by default; show it to fill fields
    await page.locator('.playbooks-editor').evaluate((el: HTMLElement) => { el.hidden = false; });

    await page.locator('#playbook-name').fill('pw-test-playbook');
    await page.locator('#playbook-description').fill('a test playbook');
    await page.locator('#playbook-body').fill('do the test thing');
    await page.locator('.playbook-save').click();

    // After save the list should re-render with the new playbook
    await expect(page.locator('.playbooks-list')).toContainText('pw-test-playbook');
  });

  test('/ menu shows a saved playbook after typing a slash', async ({ page }) => {
    // Seed the playbooks list with our test playbook
    await page.route('**/api/playbooks*', async (route) => {
      await route.fulfill({
        status: 200,
        json: { playbooks: [{ id: 1, name: 'pw-test-playbook', description: 'a test playbook', body: 'do the test thing' }] },
      });
    });

    const input = page.locator('.input-field');
    await input.fill('/pw');
    // Trigger the input event so PlaybookMenu.maybeShow() fires
    await input.dispatchEvent('input');

    await expect(page.locator('.playbook-menu')).toBeVisible();
    await expect(page.locator('.playbook-menu')).toContainText('pw-test-playbook');
  });

  test('playbooks panel opens via the agent-dropdown Playbooks button', async ({ page }) => {
    // Open the agent dropdown
    await page.locator('.agent-dropdown-btn').click();
    await expect(page.locator('.agent-dropdown-menu')).toBeVisible();

    // Click the Playbooks… button
    await page.locator('.agent-dropdown-playbooks').click();

    // Modal should now be visible
    await expect(page.locator('.playbooks-panel')).toBeVisible();

    // Close it
    await page.locator('.playbooks-close').click();
    await expect(page.locator('.playbooks-panel')).not.toBeVisible();
  });
});
