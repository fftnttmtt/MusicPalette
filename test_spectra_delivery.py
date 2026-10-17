"""Check the exact local HTTP delivery path, without loading inference models."""
import http.client
import json
from pathlib import Path
import threading
import unittest

from spectra_server import ROOT, UI_BUILD, frontend_metadata, make_available_server, make_server


class LocalDeliveryChecks(unittest.TestCase):
    def test_actual_page_and_epic_module_are_fresh_and_identifiable(self):
        server = make_server(0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        conn = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=4)
        try:
            conn.request('GET', '/api/config')
            response = conn.getresponse()
            config = json.loads(response.read())
            self.assertEqual(config['frontend']['build_id'], UI_BUILD)
            self.assertEqual(config['frontend']['epic_outline_gap_css_px'], 3)
            self.assertEqual(config['frontend']['web_directory'], str(ROOT / 'spectra_web'))
            self.assertEqual(config['project_directory'], str(ROOT))
            for path, expected in [('/?build=old', 'local-epic3-r32'),
                    ('/visuals.js?v=old', 'step=2*(width+EPIC_OUTLINE_GAP)'),
                    ('/app.js?v=old', "本地修复版 R32")]:
                conn.request('GET', path, headers={'If-Modified-Since': 'Thu, 01 Jan 2099 00:00:00 GMT'})
                response = conn.getresponse()
                body = response.read().decode('utf-8')
                self.assertEqual(response.status, 200)
                self.assertIn('no-store', response.getheader('Cache-Control'))
                self.assertEqual(response.getheader('X-Spectra-Build'), UI_BUILD)
                self.assertIn(expected, body)
        finally:
            conn.close()
            server.shutdown()
            server.server_close()
            server.jobs.executor.shutdown(wait=True)
            thread.join(2)

    def test_old_port_is_left_intact_and_new_instance_gets_another_port(self):
        old = make_server(0)
        new = None
        try:
            new = make_available_server(old.server_port)
            self.assertNotEqual(new.server_port, old.server_port)
            self.assertEqual(old.socket.getsockname()[1], old.server_port)
        finally:
            for server in (old, new):
                if server:
                    server.server_close()
                    server.jobs.executor.shutdown(wait=True)


if __name__ == '__main__':
    unittest.main()
