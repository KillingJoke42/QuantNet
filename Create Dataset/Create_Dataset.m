train_images = load('mnist.mat').trainX;

levels2bit = 1:4;
levels3bit = 1:8;
levels4bit = 1:16;
levels5bit = 1:32;
levels6bit = 1:64;
levels7bit = 1:128;
levels8bit = 1:256;

for i=1:60000
    mnist_org = reshape(train_images(i,:), 28, 28);
    imwrite(uint8(imquantize(mnist_org, levels2bit)), sprintf('C:\\Users\\Dell\\Desktop\\mnistquant%dbit\\%d.bmp', 2, i));
    imwrite(uint8(imquantize(mnist_org, levels3bit)), sprintf('C:\\Users\\Dell\\Desktop\\mnistquant%dbit\\%d.bmp', 3, i));
    imwrite(uint8(imquantize(mnist_org, levels4bit)), sprintf('C:\\Users\\Dell\\Desktop\\mnistquant%dbit\\%d.bmp', 4, i));
    imwrite(uint8(imquantize(mnist_org, levels5bit)), sprintf('C:\\Users\\Dell\\Desktop\\mnistquant%dbit\\%d.bmp', 5, i));
    imwrite(uint8(imquantize(mnist_org, levels6bit)), sprintf('C:\\Users\\Dell\\Desktop\\mnistquant%dbit\\%d.bmp', 6, i));
    imwrite(uint8(imquantize(mnist_org, levels7bit)), sprintf('C:\\Users\\Dell\\Desktop\\mnistquant%dbit\\%d.bmp', 7, i));
end