export default function(labels) {
	const zeros = Array.apply(null, Array(labels.length)).map(() => 0);
	const oneHot = labels.map((color, i) => {
		const encoded = zeros.slice();
		encoded[i] = 1;
		return encoded;
	});
	const map = {};
	labels.forEach((color, i) => {
		map[color] = oneHot[i];
	});

	return {
		oneHot,
		map,
	};
}
