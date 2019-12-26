class Matrix {
    constructor(rows = 2, cols = 2) {
        this.rows = rows;
        this.cols = cols;

        this.data = [];
        for (let i = 0; i < rows; i++) {
            this.data[i] = [];
            for (let j = 0; j < cols; j++) {
                this.data[i][j] = 0;
            }
        }
    }

    each(func) {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] = func(this.data[i][j], i, j);
            }
        }
        return this;
    }

    copy() {
        let m = new Matrix(this.rows, this.cols);
        m.each((val, i, j) => this.data[i][j]);
        return m;
    }

    log() {
        console.table(this.data);
        return this;
    }

    fix() {
        this.rows = this.data.length;
        this.cols = this.data[0].length;
        return this;
    }

    randomize(a = 0, b = 1) {
        this.each(() => (Math.random() * (b - a) + a));
        return this;
    }

    add(n) {
        if (n instanceof Matrix) {
            if (this.rows != n.rows || this.cols != n.cols) {
                console.warn("Cols and rows of A must match cols and rows of B");
            } else {
                this.each((val, i, j) => val + n.data[i][j]);
            }
        } else {
            this.each((val) => val + n);
        }
        return this;
    }

    sub(n) {
        if (n instanceof Matrix) {
            if (this.rows != n.rows || this.cols != n.cols) {
                console.warn("Cols and rows of A must match cols and rows of B");
            } else {
                this.each((val, i, j) => val - n.data[i][j]);
            }
        } else {
            this.each((val) => val - n);
        }
        return this;
    }

    set(n) {
        if (n instanceof Matrix) {
            this.set(n.data);
        } else if (n instanceof Array) {
            this.data = n;
            this.fix();
        } else {
            this.each(() => n);
        }
        return this;
    }

    dot(n) {
        if (this.cols != n.rows) {
            console.warn("Cols of A must match rows of B");
        } else {
            // https://stackoverflow.com/questions/27205018/multiply-2-matrices-in-javascript
            let m = new Matrix(this.rows, n.cols);
            m.set(0);
            for (var i = 0; i < m.rows; i++) {
                for (var j = 0; j < m.cols; j++) {
                    for (var k = 0; k < this.cols; k++) {
                        m.data[i][j] += this.data[i][k] * n.data[k][j];
                    }
                }
            }
            this.data = m.data;
            this.fix();
        }
        return this;
    }

    mult(n) {
        if (n instanceof Matrix) {
            if (this.rows != n.rows || this.cols != n.cols) {
                console.warn("Cols and rows of A must match cols and rows of B");
            } else {
                // https://en.wikipedia.org/wiki/Hadamard_product_(matrices)
                for (var i = 0; i < this.rows; i++) {
                    for (var j = 0; j < this.cols; j++) {
                        this.data[i][j] *= n.data[i][j];
                    }
                }
            }
        } else {
            this.each((val) => val * n);
        }
        return this;
    }

    transpose() {
        let m = new Matrix(this.cols, this.rows);
        m.each((val, i, j) => this.data[j][i]);
        this.set(m);
        return this;
    }

    static multiply(a, b) {
        if (a.cols != b.rows) {
            console.warn("Cols of A must match rows of B");
            return;
        }
        var m = new Matrix(a.rows, b.cols);
        m.set(0);
        for (var i = 0; i < a.rows; i++) {
            for (var j = 0; j < b.cols; j++) {
                for (var k = 0; k < a.cols; k++) {
                    m.data[i][j] += a.data[i][k] * b.data[k][j];
                }
            }
        }
        return m;
    }

    // todo refactor
    static fromArray(arr) {
        let m = new Matrix(arr.length, 1);
        for (let i = 0; i < arr.length; i++) {
            m.data[i][0] = arr[i];
        }
        return m;
    }

    static toArray(m) {
        let res = [];
        for (let i = 0; i < m.data.length; i++) {
            for (let j = 0; j < m.data[0].length; j++) {
                res.push(m.data[i][j]);
            }
        }
        return res;
    }
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function d_sigmoid(y) {
    return y * (1 - y);
}

class NeuralNetwork {
    constructor(input_nodes, hidden_nodes, output_nodes) {
        this.input_nodes = input_nodes;
        this.hidden_nodes = hidden_nodes;
        this.output_nodes = output_nodes;

        this.weights_ih = (new Matrix(this.hidden_nodes, this.input_nodes)).randomize(-1, 1);
        this.weights_ho = (new Matrix(this.output_nodes, this.hidden_nodes)).randomize(-1, 1);

        this.bias_h = (new Matrix(this.hidden_nodes, 1)).randomize(-1, 1);
        this.bias_o = (new Matrix(this.output_nodes, 1)).randomize(-1, 1);

        this.setLearningRate();

    }

    predict(input_array) {
        if (input_array.length != this.input_nodes) {
            console.error("WRONG INPUT_ARRAY LENGTH");
        }

        let inputs = Matrix.fromArray(input_array);
        let hidden = Matrix.multiply(this.weights_ih, inputs).add(this.bias_h).each(sigmoid);
        let outputs = Matrix.multiply(this.weights_ho, hidden).add(this.bias_o).each(sigmoid);

        return Matrix.toArray(outputs);
    }

    train(inputs_array, targets_array) {
        if (inputs_array.length != this.input_nodes) {
            console.error("WRONG INPUT_ARRAY LENGTH");
            return;
        } else if (targets_array.length != this.output_nodes) {
            console.error("WRONG TARGET_ARRAY LENGTH");
            return;
        }

        // Generating the Hidden Outputs
        let inputs = Matrix.fromArray(inputs_array);
        let hidden = Matrix.multiply(this.weights_ih, inputs).add(this.bias_h).each(sigmoid);
        let outputs = Matrix.multiply(this.weights_ho, hidden).add(this.bias_o).each(sigmoid);

        // Convert array to matrix object
        let targets = Matrix.fromArray(targets_array);

        // Calculate the error
        // ERROR = TARGETS - OUTPUTS
        let output_errors = targets.copy().sub(outputs);

        // let gradient = outputs * (1 - outputs);
        // Calculate gradient
        let gradients = outputs.copy().each(d_sigmoid).mult(output_errors).mult(this.learning_rate);

        let hidden_T = hidden.copy().transpose();
        let weights_ho_deltas = Matrix.multiply(gradients, hidden_T);

        // adjust the weights weithsby deltas
        this.weights_ho.add(weights_ho_deltas);
        this.bias_o.add(gradients);

        // calc hidden layer errors
        let who_t = this.weights_ho.copy().transpose();
        let hidden_errors = Matrix.multiply(who_t, output_errors);

        // calculate hidden gradient
        let hidden_gradient = hidden.copy().each(d_sigmoid).mult(hidden_errors).mult(this.learning_rate);

        // calculate input->hidden deltas
        let inputs_T = inputs.copy().transpose();
        let weights_ih_deltas = Matrix.multiply(hidden_gradient, inputs_T);

        // adjust the input->hidden weights
        this.weights_ih.add(weights_ih_deltas);
        this.bias_h.add(hidden_gradient);
    }

    setLearningRate(learning_rate = 0.1) {
        this.learning_rate = learning_rate;
    }
}