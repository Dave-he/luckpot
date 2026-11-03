import tempfile
import unittest
from pathlib import Path

from lottery.config import LOTTERY_CONFIGS
from lottery.data.loader import DataLoader
from lottery.spiders.sporttery import SportteryProvider
from lottery.spiders.cwl import CWLProvider
from lottery.spiders.data500 import Data500Provider


class PositionsTest(unittest.TestCase):
    def test_sources_and_csv_preserve_positions(self):
        cases = {'pls': [2, 7, 6], 'plw': [2, 7, 6, 2, 4],
                 'qxc': [2, 9, 4, 5, 6, 3, 12]}
        for key, numbers in cases.items():
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                payload = {'value': {'list': [{'lotteryDrawNum': '26125',
                    'lotteryDrawTime': '2026-11-01',
                    'lotteryDrawResult': ' '.join(map(str, numbers))}]}}
                records = SportteryProvider().parse_response(payload, key)
                self.assertEqual(records[0]['reds'], numbers)
                html = '<tr><td>26125</td><td class="cfont2">' + ' '.join(map(str, numbers)) + '</td></tr>'
                self.assertEqual(Data500Provider().parse_html(html, key)[0]['reds'], numbers)
                config = dict(LOTTERY_CONFIGS[key], data_file=str(Path(tmp) / 'history.csv'))
                loader = DataLoader(config)
                loader.save_data(records, append=False)
                self.assertEqual(loader.load_history()[0]['reds'], numbers)

    def test_fc3d_preserves_order(self):
        payload = {'state': 0, 'result': [{'code': '2026291', 'red': '0,8,6', 'blue': ''}]}
        self.assertEqual(CWLProvider().parse_response(payload, '3d')[0]['reds'], [0, 8, 6])

    def test_model_targets_and_outputs_keep_positions(self):
        import numpy as np
        from unittest.mock import Mock
        from lottery.models.xgboost_model import XGBoostPredictor
        from lottery.models.predictor import LotteryPredictor
        from scripts.validate_pipeline import validate_numbers
        config = LOTTERY_CONFIGS['qxc']
        numbers = [9, 1, 1, 0, 8, 3, 14]
        history = [{'reds': numbers[:], 'blues': []} for _ in range(55)]
        model = XGBoostPredictor(config)
        self.assertEqual(model._build_features(history)[1][0].tolist(), numbers)
        model.is_trained = True
        model.scaler = Mock(transform=lambda x: x)
        model.red_models = [Mock(predict=lambda x, n=n: np.array([n])) for n in numbers]
        reds, blues, _ = model.predict(history)
        self.assertEqual(reds, numbers)
        validate_numbers(config, {'reds': reds, 'blues': blues})
        traditional = LotteryPredictor(config)
        self.assertEqual(traditional.predict_by_frequency(history)[0], numbers)
        for _ in range(30):
            reds, blues = traditional.predict_random()
            validate_numbers(config, {'reds': reds, 'blues': blues})

    def test_xgboost_retrain_save_load(self):
        from lottery.models.xgboost_model import XGBoostPredictor
        from scripts.validate_pipeline import validate_numbers
        config = LOTTERY_CONFIGS['qxc']
        history = [{'reds': [9, 1, 1, 0, 8, 3, 14], 'blues': []} for _ in range(55)]
        with tempfile.TemporaryDirectory() as tmp:
            model = XGBoostPredictor(config)
            self.assertTrue(model.train(history)['success'])
            model.save(tmp)
            reloaded = XGBoostPredictor(config)
            self.assertTrue(reloaded.load(tmp))
            reds, blues, _ = reloaded.predict(history)
            self.assertEqual(reds, [9, 1, 1, 0, 8, 3, 14])
            validate_numbers(config, {'reds': reds, 'blues': blues})

    def test_prizes_require_correct_positions(self):
        from lottery.models.positional import prize_level, blend
        self.assertEqual(prize_level('pls', [2, 7, 6], [2, 6, 7]), 0)
        self.assertEqual(prize_level('plw', [2, 7, 6, 2, 4], [2, 7, 6, 2, 4]), 1)
        self.assertEqual(prize_level('qxc', [1, 2, 3, 4, 5, 6, 14], [9, 9, 9, 9, 9, 9, 14]), 6)
        self.assertEqual(blend(LOTTERY_CONFIGS['pls'], {'a': [9, 1, 1], 'b': [2, 1, 1]},
                               {'a': .8, 'b': .2}), [9, 1, 1])

    def test_update_rebuilds_sorted_history_once(self):
        from unittest.mock import patch
        from scripts.update_data import update_lottery
        import json
        with tempfile.TemporaryDirectory() as tmp:
            config = dict(LOTTERY_CONFIGS['pls'], data_file=str(Path(tmp) / 'history.csv'))
            loader = DataLoader(config)
            old = [{'issue': '26001', 'date': '', 'reds': [1, 2, 9], 'blues': []}]
            loader.save_data(old, append=False)
            restored = [{'issue': f'26{i:03d}', 'date': '', 'reds': [9, 2, 1], 'blues': []}
                        for i in range(2, 55)]
            with patch('scripts.update_data._fetch_with_fallback', return_value=(restored, 'data500')):
                result = update_lottery('pls', config)
                self.assertTrue(result['rebuilt_positions'])
                self.assertEqual(loader.load_history()[0]['reds'], [9, 2, 1])
                self.assertEqual(len(loader.load_history()), 53)
                self.assertTrue(json.loads(Path(config['data_file'] + '.schema.json').read_text())['ordered'])
                self.assertFalse(update_lottery('pls', config)['rebuilt_positions'])

    def test_lotto_still_sorts(self):
        payload = {'value': {'list': [{'lotteryDrawNum': '26125',
            'lotteryDrawResult': '29 12 23 13 25 10 07'}]}}
        record = SportteryProvider().parse_response(payload, 'dlt')[0]
        self.assertEqual(record['reds'], [12, 13, 23, 25, 29])
        self.assertEqual(record['blues'], [7, 10])


if __name__ == '__main__':
    unittest.main()
