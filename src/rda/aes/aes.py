import numpy as np

import rda.optimized.functions


def expand_keys(master_keys):

	# each master key will be spanded to 11 round keys
	rks = np.zeros((master_keys.shape[0], 11 * 16), dtype=np.uint8)

	# the first round key are the master keys
	rks[:, 0:16] = master_keys

	# round constants
	rcons = np.array([0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36], np.uint8)

	for round in range(1, 11):

		byte = round * 16

		# take the previous last 4 bytes and move the through the sbox
		tmp = rda.optimized.functions.sbox(rks[:, byte - 4:byte])

		# scramble
		tmp = tmp[:, [1, 2, 3, 0]]

		# rcon
		tmp[:, 0] ^= rcons[round]

		# copy prev key
		rks[:, byte + 0:byte + 16] = rks[:, byte - 16:byte + 0]

		# walk the key schedule
		rks[:, byte + 0:byte + 4] ^= tmp
		rks[:, byte + 4:byte + 8] ^= rks[:, byte + 0:byte + 4]
		rks[:, byte + 8:byte + 12] ^= rks[:, byte + 4:byte + 8]
		rks[:, byte + 12:byte + 16] ^= rks[:, byte + 8:byte + 12]

	return rks
