from gsdmm import GSDMM

from unittest import TestCase
from numpy import testing as nt

ut = TestCase()


def test_params():
    # get_params should return GSDMM call params
    model = GSDMM(clust=3, n_iters=10, seed=0, ngram_range=(1, 2))
    pars = model.get_params()

    ut.assertEqual(pars["token"]["ngram_range"], (1, 2))
    del pars["token"]
    args = {'clust': 3, 'alpha': 0.1, 'beta': 0.1, 'iters': 10, 'seed': 0}
    ut.assertDictEqual(pars, args)


def test_model():
    texts = [
            "apples are red",
            "apples are green",
            "the sky is really dark",
            "this is a good idea"
        ]

    model = GSDMM(seed=0)
    fit = model.fit(texts)
    pred1 = model.predict_proba(texts).argmax(1)
    pred2 = model.predict(texts)
    info = model.get_clust_info()
    imp1 = model.get_importances()
    imp2 = model.get_avg_importances()

    # text should be clustered in a, a, b, c
    ut.assertEqual(fit[0], fit[1])
    ut.assertNotEqual(fit[0], fit[2])
    ut.assertNotEqual(fit[0], fit[3])

    # predict and fit on same text must be equal
    nt.assert_array_equal(pred1, pred2)
    nt.assert_array_equal(fit, pred2)

    # should have 2 cluster with 25% of  docs and 1 with 50%
    ut.assertEqual(sum(info[:, 0] == 0.25), 2)
    ut.assertEqual(sum(info[:, 0] == 0.5), 1)

    # two importance methods should return same keys
    for i in range(10):
        ut.assertEqual(imp1[i].keys(), imp2[i].keys())


def test_wordclouds():
    texts = [
            "apples are red",
            "apples are green",
            "the sky is really dark",
            "this is a good idea"
        ]

    model = GSDMM(clust=2, seed=0)
    model.fit(texts)
    imp = model.get_avg_importances()
    fig = model.get_wordclouds(imp)

    # should return a figure
    ut.assertIsNotNone(fig)
