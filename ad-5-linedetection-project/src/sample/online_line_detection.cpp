#include <iostream>
#include <fstream>

#include "opencv2/opencv.hpp"

const int32_t gWidth = 640;
const int32_t gHeight = 480;
const int32_t gOffset = 400;
const int32_t gGap = 40;
const int32_t gHalfWidth = gWidth / 2;
const int32_t gHalfHeight = gHeight / 2;
const int32_t gLineWidth = 500;

typedef cv::Vec4f line_t;
typedef std::vector<line_t> line_vector_t;
typedef cv::Point point_t;
typedef cv::Point2f point2f_t;

/// <summary>
/// 차선을 인식하여 선을 그린다
/// </summary>
/// <param name="img">영상</param>
/// <param name="lines">라인 목록</param>
/// <param name="color">색삭</param>
void drawLines(cv::Mat& img, line_vector_t lines, cv::Scalar color);

/// <summary>
/// 차선의 영역을 인식하여 중앙의 좌표를 표시한다
/// </summary>
/// <param name="img">정지</param>
/// <param name="lpos">왼쪽 좌표</param>
/// <param name="rpos">오른쪽 좌표</param>
void drawRectangle(cv::Mat& img, float lpos, float rpos);

/// <summary>
/// 화면을 좌우를 분할하여 차선의 그룹을 나눈다
/// </summary>
/// <param name="lines">라인 목록</param>
/// <returns>좌우 라인 목록</returns>
std::pair<line_vector_t, line_vector_t> divideLeftRight(line_vector_t lines);

/// <summary>
/// 선의 m값과 b값을 계산한다
/// </summary>
/// <param name="lines">라인 목록</param>
/// <returns>파라미터값</returns>
std::pair<float, float> getLineParams(line_vector_t lines);

/// <summary>
/// 라인의 위치를 계산한다
/// </summary>
/// <param name="img">영상</param>
/// <param name="lines">라인 목록</param>
/// <param name="isLeft">왼쪽 여부</param>
/// <returns>위치 계산</returns>
int32_t getLinePosition(cv::Mat& img, line_vector_t lines, bool isLeft);

/// <summary>
/// 허프변환 알고리즘으로 영상 전처리를 한다
/// </summary>
/// <param name="frame">영상</param>
/// <returns>라인 목록</returns>
line_vector_t preprocessingImage(cv::Mat& frame);

/// <summary>
/// 영상의 차선인식을 진행한다
/// </summary>
/// <param name="frame">영상</param>
/// <returns>선의 좌우 좌표</returns>
std::pair<float, float> processImage(cv::Mat& frame);

/// <summary>
/// 히스토그램 스트레칭을 수행한다
/// </summary>
/// <param name="src_cvt">원본 영상</param>
/// <param name="src_st">변환 영상</param>
void stretchHist(cv::Mat src_cvt, cv::Mat& src_st);

/// <summary>
/// 좌우 좌표를 csv 파일에 추가로 저장한다
/// </summary>
/// <param name="filename">파일 이름</param>
/// <param name="lpos">왼쪽 좌표</param>
/// <param name="rpos">오른쪽 좌표</param>
void appendPositionData(std::string filename, float lpos, float rpos);

int main()
{
	cv::Mat frame;

	// 객체 생성 후 영상 파일 열기
	cv::VideoCapture cap;
	cap.open("Sub_project.avi");
	cap.set(cv::CAP_PROP_POS_FRAMES, 0);
	cap.set(cv::CAP_PROP_FPS, 30);

	std::string filename("output.csv");

	// 예외처리
	if (!cap.isOpened()) {
		std::cerr << "Load failed!" << std::endl;
		return -1;
	}

	int fps;
	do {
		// 객체로 부터 한 프레임을 받아서 frame 변수에 저장
		cap >> frame;

		// 이미지 처리
		std::pair<float, float> postion = processImage(frame);
		float lpos = postion.first;
		float rpos = postion.second;

		imshow("video", frame);

		fps = cap.get(cv::CAP_PROP_POS_FRAMES);
		if (fps % 30 == 0)
		{
			appendPositionData(filename, lpos, rpos);
			//std::cout << "lpos:" << lpos << " rpos:" << rpos << std::endl;
		}

		if (cv::waitKey(30) >= 0)
			continue;

	} while (fps < cap.get(cv::CAP_PROP_FRAME_COUNT));

	cap.release();
	cv::destroyAllWindows();

}

void drawLines(cv::Mat& img, line_vector_t lines, cv::Scalar color = cv::Scalar(0, 0, 255))
{
	int32_t x1, y1, x2, y2;
	for (line_t line : lines)
	{
		x1 = line[0];
		y1 = line[1];
		x2 = line[2];
		y2 = line[3];

		cv::line(img, point_t(x1, y1 + gOffset), point_t(x2, y2 + gOffset), color, 2);
	}
}

void drawRectangle(cv::Mat& img, float lpos, float rpos)
{
	float f_center = (lpos + rpos) / 2;
	f_center = round(f_center * 100) / 100;
	int32_t center = static_cast<int>(f_center);

	int32_t y1 = 15 + gOffset;
	int32_t y2 = 25 + gOffset;

	cv::rectangle(img, point_t(lpos - 5, y1), point_t(lpos + 5, y2), cv::Scalar(255, 0, 255), 2);
	cv::rectangle(img, point_t(rpos - 5, y1), point_t(rpos + 5, y2), cv::Scalar(0, 255, 255), 2);
	cv::rectangle(img, point_t(center - 5, y1), point_t(center + 5, y2), cv::Scalar(0, 255, 0), 2);
	cv::rectangle(img, point_t(gHalfWidth - 5, y1), point_t(gHalfWidth + 5, y2), cv::Scalar(0, 0, 255), 2);
}

std::pair<line_vector_t, line_vector_t> divideLeftRight(line_vector_t lines)
{
	int32_t low_slope_threshold = 0;
	int32_t high_slope_threshold = 10;
	int32_t width_threshold = 90;

	std::vector<float> slopes;
	line_vector_t new_lines;

	int32_t x1, y1, x2, y2;
	float slope, abs_slope;

	for (line_t line: lines) {
		x1 = line[0];
		y1 = line[1];
		x2 = line[2];
		y2 = line[3];

		if (x2 - x1 == 0)
		{
			slope = 0;
		}
		else
		{
			slope = static_cast<float>(y2 - y1) / static_cast<float>(x2 - x1);
		}

		abs_slope = abs(slope);
		if ((abs_slope > low_slope_threshold) && (abs_slope < high_slope_threshold)) {
			slopes.push_back(slope);
			new_lines.push_back(line);
		}
	};

	line_vector_t left_lines, right_lines;
	cv::Vec4i line;
	for (size_t i = 0; i < slopes.size(); i++) {
		line = new_lines[i];
		slope = slopes[i];
		x1 = line[0];
		y1 = line[1];
		x2 = line[2];
		y2 = line[3];

		if ((slope < 0) && (x2 < gHalfWidth - width_threshold))
		{
			left_lines.push_back(line);
		}
		else if ((slope > 0) && (x1 > gHalfWidth + width_threshold))
		{
			right_lines.push_back(line);
		}
	}

	std::pair<line_vector_t, line_vector_t> all_lines;
	all_lines.first = left_lines;
	all_lines.second = right_lines;
	return all_lines;
}

std::pair<float, float> getLineParams(line_vector_t lines)
{
	float x_sum = .0f;
	float y_sum = .0f;
	float m_sum = .0f;

	int32_t size = lines.size();
	if (size == 0)
	{
		return std::pair<float, float>(0, 0);
	}

	float x1, y1, x2, y2;
	for (line_t line : lines) {
		x1 = line[0];
		y1 = line[1];
		x2 = line[2];
		y2 = line[3];

		x_sum += x1 + x2;
		y_sum += y1 + y2;
		m_sum += static_cast<float>(y2 - y1) / static_cast<float>(x2 - x1);
	}

	float x_avg = x_sum / (size * 2);
	float y_avg = y_sum / (size * 2);
	float m = m_sum / size;
	float b = y_avg - m * x_avg;

	return std::pair<float, float>(m, b);
}

int32_t getLinePosition(cv::Mat& img, line_vector_t lines, bool isLeft = true)
{

	std::pair<float, float> params = getLineParams(lines);
	float m = params.first;
	float b = params.second;

	int32_t pos;
	if ((m == 0) && (b == 0))
	{
		pos = isLeft ? 0 : gWidth;
	}
	else
	{
		float y = gGap / 2;
		pos = static_cast<int32_t>((y - b) / m);

		b += gOffset;
		float x1 = (gHeight - b) / m;
		float x2 = (gHalfHeight - b) / m;

		cv::line(img, point_t(x1, gHeight), point_t(x2, gHalfHeight), cv::Scalar(255, 255, 0), 3);
	}
	
	pos = round(pos * 100) / 100;
	return pos;
}

std::vector<line_t> preprocessingImage(cv::Mat& frame)
{
	// Gray
	cv::Mat src_cvt, src_st, src_blur, src_edge, src_mask, roi;
	
	// Gray
	cvtColor(frame, src_cvt, cv::COLOR_BGR2GRAY);

	// Stretch Histogram 적용, Histogram Equalize는 효과가 없었음(노이즈 증가)
	stretchHist(src_cvt, src_st);

	// Gaussian blur, sigmaX는 노이즈에 효과적
	GaussianBlur(src_st, src_blur, cv::Size(), 1.0);

	// Canny edge, threshold1과 threshold2의 비율은 1:2 또는 1:3을 권장, L2 norm 고려해보기
	Canny(src_blur, src_edge, 100, 300);

	// ROI
	roi = src_edge(cv::Rect(0, gOffset, gWidth, gGap));
	cv::rectangle(src_edge, cv::Rect(220, 400, 200, 50), cv::Scalar(0, 0, 0), -1);

	// HoughLinesP
	std::vector<line_t> lines;
	HoughLinesP(roi, lines, 1, CV_PI / 180, 30, 20, 3);
	
	return lines;
}

void stretchHist(cv::Mat src_cvt, cv::Mat& src_st) {
	double gmin, gmax;
	minMaxLoc(src_cvt, &gmin, &gmax);
	src_st = (src_cvt - gmin) * 255 / (gmax - gmin);
}

std::pair<float, float> processImage(cv::Mat& frame)
{
	point_t pt;
	line_vector_t all_lines = preprocessingImage(frame);
	if (all_lines.empty()) {
		// 차선이 없으면 마지막에 인식한 좌표 사용
		return std::pair<float, float>(0, gWidth);
	};

	std::pair<line_vector_t, line_vector_t> pair_lines;
	pair_lines = divideLeftRight(all_lines);
	line_vector_t left_lines = pair_lines.first;
	line_vector_t right_lines = pair_lines.second;

	float lpos, rpos;
	lpos = getLinePosition(frame, left_lines, true);
	rpos = getLinePosition(frame, right_lines, false);

	int32_t interval = 10;
	if (left_lines.size() == 0 && right_lines.size() > 0)
	{
		// 왼쪽 차선 미인식
		lpos = rpos - gLineWidth;
	}
	else if (left_lines.size() > 0 && right_lines.size() == 0) {
		// 오른쪽 차선 미인식
		rpos = lpos + gLineWidth;
	}

	drawLines(frame, left_lines, cv::Scalar(255, 0, 255));
	drawLines(frame, right_lines, cv::Scalar(0, 255, 255));

	drawRectangle(frame, lpos, rpos);

	return std::pair<float, float>(lpos, rpos);
}

void appendPositionData(std::string filename, float lpos, float rpos)
{
	std::fstream output_stream;

	output_stream.open(filename, std::ios::out | std::ios::app);
	if (!output_stream.is_open())
	{
		std::cerr << "failed to open " << filename << std::endl;
	}
	else
	{
		output_stream << lpos << "," << rpos << std::endl;
	}
}
