# Monte Carlo tournament simulation.
#
# Takes per-match probabilities from the chosen model, samples outcomes, and
# propagates winners through the WC26 bracket (group stage + knockouts) over
# >=10k iterations. Knockout progression is framed as an absorbing Markov chain.
# Reports per-team stage-reach and championship probabilities.
