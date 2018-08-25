import React, {Component} from 'react';
import * as tf from '@tensorflow/tfjs';
import oneHotEncode from '../../utils/oneHotEncode';
import './app.css';

// one hot encoded labels
const COLOR_LABELS = [
	'red',
	'green',
	'blue',
	'black',
	'white',
	'orange',
	'pink',
	'purple',
	'yellow',
];
const ONE_HOT_ENCODED = oneHotEncode(COLOR_LABELS);
const COLORS = ONE_HOT_ENCODED.map;

const xs_raw = [
	[0.1, 1.0, 0.05],
	[0.95, 0.0, 0.02],
	[0.08, 0.1, 1],
	[0.0, 0.0, 1],
	[1, 0.0, 0.0],
	[0, 1.0, 0.0],
	[0, 0.0, 0.0],
	[1, 1, 1],
	[1, 0.5, 0],
	[1, 0, 0.5],
	[0.5, 0, 1],
	[1, 1, 0],
];
const xs = tf.tensor2d(xs_raw);
const ys = tf.tensor2d([
	COLORS.green,
	COLORS.red,
	COLORS.blue,
	COLORS.blue,
	COLORS.red,
	COLORS.green,
	COLORS.black,
	COLORS.white,
	COLORS.orange,
	COLORS.pink,
	COLORS.purple,
	COLORS.yellow,
]);

class App extends Component {
	constructor() {
		super();
		this.state = {
			currentPrediction: '',
			hasTrained: false,
			trainingProgres: 0,
			trainingLoss: 1,
		};

		this.handleInput = this.handleInput.bind(this);
	}

	componentDidMount() {
		this.initModel();
	}

	initModel() {
		this.model = tf.sequential();

		const hiddenLayer = tf.layers.dense({
			units: 20,
			inputShape: [3],
			activation: 'relu',
		});
		this.model.add(hiddenLayer);

		const outputLayer = tf.layers.dense({
			units: COLOR_LABELS.length,
			activation: 'sigmoid',
		});
		this.model.add(outputLayer);

		this.model.compile({
			optimizer: tf.train.adam(0.035),
			loss: 'meanSquaredError',
		});

		this.train().then(() => {
			this.setState({
				hasTrained: true,
			});
		});
	}

	async train() {
		const rounds = 100;
		for (let i = 0; i < rounds; i++) {
			console.log(`--\nTraining round ${i}`);
			await this.model.fit(xs, ys, {
				ephocs: 250,
				batchSize: 1,
				shuffle: true,
				callbacks: {
					onEpochEnd: async (epoch, log) => {
						this.setState({
							trainingLoss: log.loss,
						});
						console.log(`Loss = ${log.loss}`);
					},
				},
			});
			this.setState({
				trainingProgres: (i / rounds) * 100,
			});
		}
	}

	async predict(normalizedRgb) {
		const prediction = this.model.predict(tf.tensor2d([normalizedRgb]));
		prediction.print();

		const [predictedColor] = await Promise.all([
			prediction.argMax(1).data(),
		]);

		return COLOR_LABELS[predictedColor];
	}

	handleInput(event) {
		const hex = event.target.value.replace('#', '');
		const hexArray = hex.match(/.{1,2}/g);
		const normalizedDecArray = hexArray.map(hex => {
			return parseInt(hex, 16) / 255;
		});
		this.predict(normalizedDecArray).then(color => {
			console.log(color);
			this.setState({
				currentPrediction: color,
			});
		});
	}

	render() {
		const {
			currentPrediction,
			hasTrained,
			trainingProgres,
			trainingLoss,
		} = this.state;
		return (
			<div className="App">
				{!hasTrained ? (
					<div>
						Currently training model:{' '}
						{parseInt(trainingProgres, 10)}
						%, loss: {trainingLoss}
					</div>
				) : (
					<div>
						Training done! Choose a color{' '}
						<div>
							<input
								className="color-input"
								type="color"
								onChange={this.handleInput}
							/>
						</div>
						{currentPrediction && (
							<div>
								The current prediction is {currentPrediction}
							</div>
						)}
						<div />
					</div>
				)}
			</div>
		);
	}
}

export default App;
