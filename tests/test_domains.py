import unittest
from tools.evaluate_domains import domain_counts, canonical_domain


class DomainTests(unittest.TestCase):
    def test_domain_counts_separate_relevance_and_allow_cross_domain_support(self):
        chunks=[{'document_id':d} for d in ['A','A','B','C']]
        result=domain_counts(chunks,'AI Papers',{'A':'AI Papers','B':'AI Papers','C':'DONES'}, {'A':0,'B':2,'C':1})
        self.assertEqual(result['same_domain_chunks'],3)
        self.assertEqual(result['same_domain_fraction'],.75)
        self.assertEqual(result['same_domain_grade0_chunks'],2)
        self.assertEqual(result['other_domain_grade1_or2_chunks'],1)

    def test_empty_and_unknown_domains(self):
        self.assertIsNone(domain_counts([],'DONES',{}, {})['same_domain_fraction'])
        with self.assertRaises(ValueError):canonical_domain('unknown')
