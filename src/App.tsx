import { map } from 'lodash';
import React, { useEffect, useState } from 'react';
import './App.css';
import * as tfvis from '@tensorflow/tfjs-vis';
import * as tf from '@tensorflow/tfjs';

function App() {
  const [trainDataSource, setTrainDataSource] = useState<any[]>([]);

  // 1. 获取训练数据源
  const fetchTrainData = async () => {
    const dataRes = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
    const data = await dataRes.json();
    
    setTrainDataSource(
      map(data, item => ({
        mpg: item.Miles_per_Gallon,
        horsepower: item.Horsepower
      })).filter(item => item.mpg && item.horsepower)
    );
  }

  // 1.1 可视化展示训练数据源，用于体现参数之间的关系
  const renderTrainDataSource = () => {
    const chartData = map(trainDataSource, item => ({ x: item.horsepower, y: item.mpg }));
    // 散点图展示
    tfvis.render.scatterplot({name: '马力/单位油耗英里数'}, { values: chartData }, {
      xLabel: '马力',
      yLabel: '单位油耗英里数',
      height: 300
    })
  }

  // 2. 初始化模型
  const createModel = () => {
    const model = tf.sequential();
    /**
     *  神经网络的核心组成部分是层，这是数据处理模块，你可以把它看作是一个从张量到张量的可调函数。
     *  dense 是一种层，可将输入与矩阵（称为“权重”）相乘，并向结果添加一个数字（称为“偏差”）。
     *  由于这是网络的第一层，因此我们需要定义 inputShape。inputShape 是 [1]，因为我们将 1 数字用作输入（训练数据中的 horsepower）。
     *  units 用于设置权重矩阵在层中的大小。将其设置为 1 即表示数据的每个输入特征的权重为 1。
     */
    // 第一层隐藏层
    model.add(tf.layers.dense({ inputShape: [1], units: 1 }));
    // 在第一隐藏层和最终输出层之间添加更多的隐藏层
    model.add(tf.layers.dense({ units: 50, activation: 'sigmoid' }));

    // 输出层
    model.add(tf.layers.dense({ units: 1 }));
    tfvis.show.modelSummary({ name: '模型概要' }, model);

    return model;
  }

  // 3. 重排 & 归一化数据
  const convertToTensor = (data: any[]) => {
    return tf.tidy(() => {
      // 1. 重排数据
      /**
       * 数据重排很重要，因为在训练期间，数据集通常会被拆分成较小的子集（称为批次），以用于训练模型。借助重排，每个批次可从分布的所有数据中获取各种数据。通过这样做，我们可以帮助模型：
       * 1. 不学习纯粹依赖于数据输入顺序的东西
       * 2. 对子组中的结构不敏感（例如，如果模型在训练的前半部分仅看到高位值，可能会学习一种不适用于数据集其余部分的关系）。
       */
      tf.util.shuffle(data);

      // 2. 转换为张量
      // 输入样本
      const inputs = data.map(item => item.horsepower);
      // 真实输出值（机器学习中称为标签）
      const labels = data.map(item => item.mpg);
      const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
      const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

      // 3. 归一化
      /**
       * 使用最小-最大缩放比例将数据归一化为数值范围 0-1。
       * 归一化至关重要，因为您将使用 tensorflow.js 构建的许多机器学习模型的内部构件旨在处理不太大的数字。
       * 对数据进行归一化以包含 0 to 1 或 -1 to 1 的通用范围。
       * 如果您养成将数据归一化到某合理范围内的习惯，那么在训练模型时就更有可能取得成功。
       */
      const inputMax = inputTensor.max();
      const inputMin = inputTensor.min();
      const labelMax = labelTensor.max();
      const labelMin = labelTensor.min();

      const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
      const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

      // 4. 返回处理后的数据和归一化边界
      /**
       * 我们希望保留训练期间用于归一化的值，以便我们可以将输出取消归一化，以使其恢复到原始比例。
       * 并且使我们能以相同方式对今后的输入数据进行归一化。
       */
      return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        inputMax,
        inputMin,
        labelMax,
        labelMin
      };
    })
  }

  // 训练模型
  const trainModel = async (model: tf.Sequential, inputs: any, labels: any) => {
    // 为训练做准备：编译
    /**
     * 在我们训练模型之前，我们必须对其进行“编译”。为此，我们必须指定一些非常重要的事项：
     * - optimizer：这是用于控制模型更新的算法，如样本所示。TensorFlow.js 中提供了许多优化器。我们选择了 Adam 优化器，因为它在实际使用中非常有效，无需进行任何配置。
     * - loss：这是一个函数，用于告知模型在学习所显示的各个批次（数据子集）时的表现如何。我们使用 meanSquaredError 将模型所做的预测与真实值进行比较。
     */
    model.compile({
      optimizer: tf.train.adam(),
      loss: tf.losses.meanSquaredError,
      metrics: ['mse']
    });

    // 定义 batchSize 和多少个周期
    /**
     * batchSize 是指模型在每次训练迭代时会看到的数据子集的大小。常见的批次大小通常介于 32-512 之间。对于所有问题，实际上并没有理想的批次大小。
     * epochs 表示模型查看您提供的整个数据集的次数。我们将对数据集执行 50 次迭代。
     */
    const batchSize = 32;
    const epochs = 100;

    // 启动训练循环
    /**
     * 是您为了启动训练循环而调用的函数。
     * 为了监控训练进度，我们会将一些回调传递给 model.fit。使用 tfvis.show.fitCallbacks 来生成可为之前指定的“损失”和“均方误差'”指标绘制图表的函数。
     */
    return await model.fit(inputs, labels, {
      batchSize,
      epochs,
      shuffle: true,
      callbacks: tfvis.show.fitCallbacks({ name: '训练表现' }, ['loss', 'mse'], { height: 200, callbacks: ['onEpochEnd'] })
    });
  }

  // 测试模型
  const testModel = (model: tf.Sequential, normalizationData: any) => {
    const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

    const [xs, preds] = tf.tidy(() => {
      const xs = tf.linspace(0, 1, 100);
      /**
       * 生成了 100 个新 “样本”，以提供给模型。
       * Model.predict 是我们将这些样本提供给模型的方式。
       * 请注意，它们必须具有与训练时相似的形状 ([num_examples, num_features_per_example])。
       */
      const preds = model.predict(xs.reshape([100, 1])) as any;
      preds.print();

      // 做归一化时的逆运算
      const unNormXs = xs
        .mul(inputMax.sub(inputMin))
        .add(inputMin);

      const unNormPreds = preds
        .mul(labelMax.sub(labelMin))
        .add(labelMin);

      /**
       * .dataSync() 是一种用于获取张量中存储的值的 typedarray 的方法，是通常首选的 .data() 方法的同步版本
       * 这使我们能够在常规 JavaScript 中处理这些值。
       */
      return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });

    const predictedPoints = Array.from(xs).map((val, i) => {
      return {x: val, y: preds[i]}
    });

    const originalPoints = trainDataSource.map(d => ({
      x: d.horsepower, y: d.mpg,
    }));

    tfvis.render.scatterplot(
      {name: '模型预测结果 & 原始数据'},
      {values: [originalPoints, predictedPoints], series: ['原始数据', '预测结果']},
      {
        xLabel: '马力',
        yLabel: '单位油耗英里数',
        height: 300
      }
    );
  }

  const runTrainModel = async () => {
    const model = createModel();
    const { inputs, labels, ...rest } = convertToTensor(trainDataSource);

    await trainModel(model, inputs, labels);
    console.log('--- Done Training');
    testModel(model, rest);
    console.log('--- Done Testing');
  }

  useEffect(() => {
    fetchTrainData();
  }, []);

  return (
    <div className="app">
      <header className="app-header">header</header>
      <div className="app-layout">
        <div className="app-sider">
          sider
        </div>
        <div className="app-main">
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <button onClick={renderTrainDataSource}>训练数据散点图</button>
            <button onClick={createModel}>创建模型</button>
            <button onClick={runTrainModel}>开始训练模型</button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
