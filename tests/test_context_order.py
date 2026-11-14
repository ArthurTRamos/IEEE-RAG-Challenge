import copy
import unittest

from tools.compare_context_order import order_selected


class ContextOrderTests(unittest.TestCase):
    def test_permutation_preserves_selected_text_and_never_adds_candidates(self):
        selected = [{'chunk_id':'a', 'text':'partial text', 'truncated':True},
                    {'chunk_id':'b', 'text':'full text', 'truncated':False}]
        before = copy.deepcopy(selected)
        ranking = [{'chunk_id':'b'}, {'chunk_id':'outside'}, {'chunk_id':'a'}]
        result = order_selected(selected, ranking)
        self.assertEqual(result, [before[1], before[0]])
        self.assertEqual(selected, before)
        result[0]['text'] = 'changed'
        self.assertEqual(selected, before)

    def test_invalid_catalogs_are_rejected(self):
        for selected, ranking in [([{'chunk_id':'a'}], [{'chunk_id':'b'}]),
                                  ([{'chunk_id':'a'}]*2, [{'chunk_id':'a'}]),
                                  ([{'chunk_id':'a'}], [{'chunk_id':'a'}]*2)]:
            with self.assertRaises(ValueError):
                order_selected(selected, ranking)
