#include <depthai/depthai.hpp>

int main(int argc, char **argv)
{
    auto nnPath = std::string(argv[1]);
    std::cout << "Using blob at path: " << nnPath.c_str() << std::endl;

    dai::Pipeline pipeline;

    auto colorCam = pipeline.create<dai::node::ColorCamera>();
    auto imageEncoder = pipeline.create<dai::node::NeuralNetwork>();

    auto xoutPreview = pipeline.create<dai::node::XLinkOut>();
    auto xoutEmbedding = pipeline.create<dai::node::XLinkOut>();

    xoutPreview->setStreamName("preview");
    xoutEmbedding->setStreamName("embedding");

    colorCam->setBoardSocket(dai::CameraBoardSocket::CAM_A);
    colorCam->setColorOrder(dai::ColorCameraProperties::ColorOrder::RGB);
    colorCam->setInterleaved(false);
    colorCam->setPreviewSize(1024, 1024);
    colorCam->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
    colorCam->setFps(2);
    colorCam->setPreviewKeepAspectRatio(false);

    imageEncoder->setBlobPath(nnPath);
    imageEncoder->setNumInferenceThreads(2);
    imageEncoder->setNumNCEPerInferenceThread(1);
    imageEncoder->input.setBlocking(false);

    colorCam->preview.link(imageEncoder->input);
    imageEncoder->passthrough.link(xoutPreview->input);
    imageEncoder->out.link(xoutEmbedding->input);

    dai::Device device(pipeline);

    auto previewQueue = device.getOutputQueue("preview", 4, false);
    auto embeddingQueue = device.getOutputQueue("embedding", 4, false);

    while (true)
    {
        auto image_embedding = embeddingQueue->get<dai::NNData>();
        cv::imshow("preview", previewQueue->get<dai::ImgFrame>()->getCvFrame());

        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q')
        {
            device.close();
            return 0;
        }
    }

    return 0;
}