// Copyright (C) 2023 Grepp CO.
// All rights reserved.

/**
 * @file HoughTransformLaneDetector.cpp
 * @author Jongrok Lee (lrrghdrh@naver.com)
 * @author Jiho Han
 * @author Haeryong Lim
 * @author Chihyeon Lee
 * @brief hough transform lane detector class source file
 * @version 1.1
 * @date 2023-05-02
 */

#include <numeric>

#include "LaneKeepingSystem/HoughTransformLaneDetector.hpp"

namespace Xycar {

template <typename PREC>
void HoughTransformLaneDetector<PREC>::setConfiguration(const YAML::Node& config)
{
    mImageWidth = config["IMAGE"]["WIDTH"].as<int32_t>();
    mImageHeight = config["IMAGE"]["HEIGHT"].as<int32_t>();
    mROIStartHeight = config["IMAGE"]["ROI_START_HEIGHT"].as<int32_t>();
    mROIHeight = config["IMAGE"]["ROI_HEIGHT"].as<int32_t>();
    mCannyEdgeLowThreshold = config["CANNY"]["LOW_THRESHOLD"].as<int32_t>();
    mCannyEdgeHighThreshold = config["CANNY"]["HIGH_THRESHOLD"].as<int32_t>();
    mHoughLineSlopeRange = config["HOUGH"]["ABS_SLOPE_RANGE"].as<PREC>();
    mHoughThreshold = config["HOUGH"]["THRESHOLD"].as<int32_t>();
    mHoughMinLineLength = config["HOUGH"]["MIN_LINE_LENGTH"].as<int32_t>();
    mHoughMaxLineGap = config["HOUGH"]["MAX_LINE_GAP"].as<int32_t>();
    mYAxisMargin = config["HOUGH"]["Y_AXIS_MARGIN"].as<int32_t>();
    mDebugging = config["DEBUG"].as<bool>();
    mSaveImage = config["SAVE"]["ENABLE"].as<bool>();
    mSaveImagePath = config["SAVE"]["PATH"].as<std::string>();

    mRoi = cv::Rect(cv::Point(0,mROIStartHeight), cv::Point(mImageWidth, mROIStartHeight+mROIHeight));
}

template <typename PREC>
std::pair<PREC, PREC> HoughTransformLaneDetector<PREC>::getLineParameters(const Lines& lines, const Indices& lineIndices)
{
    PREC m = 0.0F;
    PREC b = 0.0F;

    PREC x_sum = .0F;
    PREC y_sum = .0F;
    PREC m_sum = .0F;

    int32_t size = lineIndices.size();
    if (size == 0)
    {
        return std::pair<PREC, PREC>(0, 0);
    }

    PREC x1, y1, x2, y2;
    for (int32_t i : lineIndices) {
        l = lines[i];
        x1 = l[0];
        y1 = l[1];
        x2 = l[2];
        y2 = l[3];

        x_sum += x1 + x2;
        y_sum += y1 + y2;
        m_sum += static_cast<PREC>(y2 - y1) / static_cast<PREC>(x2 - x1);
    }

    PREC x_avg = x_sum / (size * 2);
    PREC y_avg = y_sum / (size * 2);
    m = m_sum / size;
    b = y_avg - m * x_avg;

    return { m, b };
}

template <typename PREC>
int32_t HoughTransformLaneDetector<PREC>::getLinePositionX(const Lines& lines, const Indices& lineIndices, Direction direction)
{
    int32_t positionX = 0;

    auto [m, b] = getLineParameters(lines, lineIndices);

    if((fabsf(m) < std::numeric_limits<PREC>::epsilon()) && (fabsf(b) < std::numeric_limits<PREC>::epsilon()))
    {
        positionX = (direction == Direction::LEFT) ? 0 : mImageWidth;
    }
    else
    {
        int32_t y = static_cast<int32_t>(mROIHeight / 2);
        positionX = static_cast<int32_t>((y - b) / m);
        // b += mROIStartHeight;
    }

    return positionX;
}

template <typename PREC>
void HoughTransformLaneDetector<PREC>::getTrackPointX(int32_t positionX, int32_t& trackPosX, float areaMin, float areaMax){
    if((areaMin <= positionX) && (positionX <= areaMax)){
        trackPosX = positionX;
        return ;
    }

}

template <typename PREC>
std::pair<Indices, Indices> HoughTransformLaneDetector<PREC>::divideLines(const Lines& lines)
{
    Indices leftLineIndices;
    Indices rightLineIndices;

    int32_t x1;
    int32_t y1;
    int32_t x2;
    int32_t y2;
    PREC slope;

    cv::Vec4f l;
    for (size_t i = 0; i < lines.size(); i++) {
        l = lines[i];
        
        // 0 1 2 3 x1 y1 x2 y2
        x1 = l[HoughIndex::x1]; 
        y1 = l[HoughIndex::y1];
        // point of start(top)
        x2 = l[HoughIndex::x2]; 
        y2 = l[HoughIndex::y2];
        // point of end(bottom)

        if(y1 > y2){
            std::swap(y1, y2);
            std::swap(x1, x2);
        }

        // 시작점 필터링
        if (y2 < mROIHeight - mYAxisMargin) continue;

        if(x1 - x2 == 0 || y1 - y2 == 0){
            slope = 0;
        }else{
            slope = static_cast<PREC>(y2 - y1) / static_cast<PREC>(x2 - x1);
        } 

        // if(slope == 0 || abs(slope) < 0.2) continue; 
        if ((fabsf(slope) < std::numeric_limits<PREC>::epsilon()) || abs(slope) < 0.2) continue; 

        // 평행선 filtering
        if(x2 < (mImageWidth/2)){
            // case is left
            leftLineIndices.push_back(i);
        }else{
            // case is right
            rightLineIndices.push_back(i);
        }
    }
            
    return { leftLineIndices, rightLineIndices };
}

// 향후 클래스로 변경 
double calcMedian(cv::Mat src){
    std::vector<double> src_vec;
    src = src.reshape(0,1);
    src.copyTo(src_vec, cv::noArray());
    std::nth_element(src_vec.begin(), src_vec.begin() + src_vec.size() / 2, src_vec.end());
    return src_vec[src_vec.size() / 2];
}

cv::Mat canny(cv::Mat src, float sigma=0.33){
    double median = calcMedian(src);
    int32_t lower;
    int32_t upper;

    lower = (int)std::max(0.0  , (1.0-sigma) * median);
    upper = (int)std::min(255.0, (1.0+sigma) * median);
    cv::Canny(src, src, lower, upper);

    return src;
}


int32_t index = 0;
template <typename PREC>
std::pair<int32_t, int32_t> HoughTransformLaneDetector<PREC>::getLanePosition(const cv::Mat& frame)
{
    static constexpr int32_t img_thr = 100;
    static constexpr int32_t img_thr_max = 255;
    static constexpr int32_t gaussian_filter = 3;
    static constexpr int32_t gaussian_sigma = 1;
    static constexpr int32_t mean_brightness = 128;

    // static constexpr int32_t img_thr = 100;
    // static constexpr int32_t img_thr_max = 255;
    // static constexpr int32_t gaussian_filter = 3;
    // static constexpr int32_t gaussian_sigma = 2;
    static constexpr int32_t hough_rho = 1;
    static constexpr int32_t hough_min = 20;
    static constexpr int32_t hough_max_gab = 0;
    int32_t leftPositionX = 0;
    int32_t rightPositionX = 0;
    
#if 0
    cv::cvtColor(frame, frame_roi, cv::COLOR_BGR2HSV);
    cv::split(frame_roi, planes);

    frame_roi = planes[2];

    // ROI crop
    frame_roi = frame_roi(mRoi);

    // blur
    cv::GaussianBlur(frame_roi, frame_roi, cv::Size(gaussian_filter, gaussian_filter), gaussian_sigma);

    // bright filter
    cv::add(frame_roi, (128 - cv::mean(frame_roi, cv::noArray())[0]), frame_roi);

    
    // binarization
    cv::threshold(frame_roi, frame_roi, img_thr, img_thr_max, cv::THRESH_BINARY_INV);

    // edge    
    frame_roi=canny(frame_roi);
#else

    cv::Mat hist, src, dst, bin, roi;

    cv::cvtColor(frame, frame_roi, cv::COLOR_BGR2HSV);
    cv::split(frame_roi, planes);

    src = planes[2];
    frame_roi = src(mRoi).clone();
    cv::blur(frame_roi, frame_roi, cv::Size(3, 3));

    int roi_mean = cv::mean(frame_roi)[0];
    frame_roi = frame_roi + (128 - roi_mean);

    int32_t gmin = 50, gmax = 160;
    frame_roi = (frame_roi - gmin) * 255 / (gmax - gmin);
    cv::dilate(frame_roi, frame_roi, cv::Mat(), cv::Point(-1,-1), 3);
    cv::erode(frame_roi, frame_roi, cv::Mat(), cv::Point(-1,-1), 3);


    // hist = getGrayHistImage(calcGrayHist(src));
    // cv::imshow("srchist", hist);
    // hist = getGrayHistImage(calcGrayHist(roi));
    // cv::imshow("roihist", hist);

    cv::threshold(frame_roi, frame_roi, 90, 255, cv::THRESH_BINARY_INV);

    frame_roi=canny(frame_roi);


#endif

    // hough
    cv::HoughLinesP(frame_roi, lines, hough_rho, CV_PI / 180, hough_min, hough_max_gab, mROIHeight-mYAxisMargin);
    cv::cvtColor(frame_roi, frame_roi, cv::COLOR_GRAY2BGR);

    auto [hLeftLineIndices, hRightLineIndices] = divideLines(lines);
    leftPositionX = getLinePositionX(lines, hLeftLineIndices, Direction::LEFT);
    rightPositionX = getLinePositionX(lines, hRightLineIndices, Direction::RIGHT);

    getTrackPointX(leftPositionX, leftTrackPosx, 5, mImageWidth*0.25+50);
    getTrackPointX(rightPositionX, rightTrackPosx, mImageWidth*0.75-50, mImageWidth-5);


	if (mDebugging || mSaveImage)
	{
        frame.copyTo(mDebugFrame);
		drawLines(lines, hLeftLineIndices, hRightLineIndices);
	}

    

    return { leftTrackPosx * 1.1f, rightTrackPosx * 1.1f };
}

template <typename PREC>
void HoughTransformLaneDetector<PREC>::drawLines(const Lines& lines, const Indices& leftLineIndices, const Indices& rightLineIndices)
{
    auto draw = [this](const Lines& lines, const Indices& indices) {
        for (const auto index : indices)
        {
            const auto& line = lines[index];
            auto r = static_cast<PREC>(std::rand()) / RAND_MAX * std::numeric_limits<uint8_t>::max();
            auto g = static_cast<PREC>(std::rand()) / RAND_MAX * std::numeric_limits<uint8_t>::max();
            auto b = static_cast<PREC>(std::rand()) / RAND_MAX * std::numeric_limits<uint8_t>::max();

            cv::line(mDebugFrame, { line[static_cast<uint8_t>(HoughIndex::x1)], line[static_cast<uint8_t>(HoughIndex::y1)] + mROIStartHeight },
                     { line[static_cast<uint8_t>(HoughIndex::x2)], line[static_cast<uint8_t>(HoughIndex::y2)] + mROIStartHeight }, { b, g, r }, kDebugLineWidth);
        }
    };

    draw(lines, leftLineIndices);
    draw(lines, rightLineIndices);
}

template <typename PREC>
void HoughTransformLaneDetector<PREC>::drawRectangles(int32_t leftPositionX, int32_t rightPositionX, int32_t estimatedPositionX)
{
    cv::rectangle(mDebugFrame, cv::Point(leftPositionX - kDebugRectangleHalfWidth, kDebugRectangleStartHeight + mROIStartHeight),
                  cv::Point(leftPositionX + kDebugRectangleHalfWidth, kDebugRectangleEndHeight + mROIStartHeight), kGreen, kDebugLineWidth);

    cv::rectangle(mDebugFrame, cv::Point(rightPositionX - kDebugRectangleHalfWidth, kDebugRectangleStartHeight + mROIStartHeight),
                  cv::Point(rightPositionX + kDebugRectangleHalfWidth, kDebugRectangleEndHeight + mROIStartHeight), kGreen, kDebugLineWidth);

    cv::rectangle(mDebugFrame, cv::Point(estimatedPositionX - kDebugRectangleHalfWidth, kDebugRectangleStartHeight + mROIStartHeight),
                  cv::Point(estimatedPositionX + kDebugRectangleHalfWidth, kDebugRectangleEndHeight + mROIStartHeight), kRed, kDebugLineWidth);

    cv::rectangle(mDebugFrame, cv::Point(mImageWidth / 2 - kDebugRectangleHalfWidth, kDebugRectangleStartHeight + mROIStartHeight),
                  cv::Point(mImageWidth / 2 + kDebugRectangleHalfWidth, kDebugRectangleEndHeight + mROIStartHeight), kBlue, kDebugLineWidth);
}

template class HoughTransformLaneDetector<float>;
template class HoughTransformLaneDetector<double>;
} // namespace Xycar