
package textscan;

import java.util.stream.*;
import java.util.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.*;
import org.opencv.imgproc.*;

public class TextScan {

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        TextScan scanner = new TextScan();
        scanner.rectDetection();
    }
    
    private void rectDetection() {
        Mat original = Imgcodecs.imread("images/capturaReal3.png");

        Mat gray = new Mat();
        Imgproc.cvtColor(original, gray, Imgproc.COLOR_BGR2GRAY);

        CLAHE clahe = Imgproc.createCLAHE(2.0, new Size(8, 8));
        Mat contrastEnhanced = new Mat();
        clahe.apply(gray, contrastEnhanced);

        Mat binary = new Mat();
        Imgproc.threshold(contrastEnhanced, binary, 100, 255, Imgproc.THRESH_TRIANGLE);
        
        Mat invertedImage = new Mat(binary.rows(), binary.cols(), CvType.CV_8UC1);
        Core.bitwise_not(binary, invertedImage);        
        
        
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(invertedImage, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        List<Rect> rectangles = new ArrayList<>();
        for (MatOfPoint contour : contours) {
            Rect rect = Imgproc.boundingRect(contour);
            rectangles.add(rect);
        }

        rectangles.sort((r1, r2) -> Double.compare(r2.area(), r1.area()));
        
        Set<Rect> uniqueRects = new LinkedHashSet<>(rectangles);
        
        List<Rect> filteredRectangles = new ArrayList<>(uniqueRects);

        Rect largestRect = Collections.max(filteredRectangles, Comparator.comparingDouble(Rect::area));
        filteredRectangles.remove(largestRect);
        
        List<Rect> top5Rectangles = filteredRectangles.stream().limit(6).collect(Collectors.toList());
        
        for (Rect rect : top5Rectangles) {
            Imgproc.rectangle(original, rect.tl(), rect.br(), new Scalar(0, 255, 0), 2);
        }
        
        for (int i = 1; i < 5; i++) {
            if (top5Rectangles.size() > 1) {
                Rect subRect = top5Rectangles.get(i);
                Mat cropped = invertedImage.submat(subRect);

                answerDetection(cropped, original);
            }
        }
        
        Imgcodecs.imwrite("output.png", original);
    }
        
    int globalCounter = 1;
    
    private void answerDetection(Mat cropped, Mat original) {
        Object[][] answers = new Object[0][0];
        
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();

        Imgproc.findContours(cropped, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        List<Rect> rectangles = contours.stream()
            .map(Imgproc::boundingRect)
            .filter(rect -> rect.width > rect.height && rect.width > rect.height * 2)
            .sorted(Comparator.comparingDouble(rect -> -rect.area()))
            .collect(Collectors.toList());
        
        rectangles.sort((r1, r2) -> {
            if (Math.abs(r1.y - r2.y) > 10) {
                return Integer.compare(r1.y, r2.y);
            }
            return Integer.compare(r1.x, r2.x);
        });

        List<Rect> topRectangles = rectangles.stream().limit(11).collect(Collectors.toList());

        int circleNumber = 0;
                
        for (Rect rect : topRectangles) {
            Imgproc.rectangle(original, rect.tl(), rect.br(), new Scalar(255, 0, 0), 2);

            Mat roi = cropped.submat(rect);

            Mat circles = new Mat();
            Imgproc.HoughCircles(roi, circles, Imgproc.HOUGH_GRADIENT, 1.3, 20, 100, 20, 30, 38);

            List<Circle> detectedCircles = new ArrayList<>();
            
            for (int i = 0; i < circles.cols(); i++) {
                double[] circle = circles.get(0, i);
                int x = (int) circle[0] + rect.x;
                int y = (int) circle[1] + rect.y;
                int radius = (int) circle[2];

                detectedCircles.add(new Circle(x, y, radius, false));
                
            }

            List<Circle> uniqueCircles = filterOverlappingCircles(detectedCircles);
            
            List<Rect> detectedRect = new ArrayList<>();
            
            uniqueCircles.sort(Comparator.comparingInt(c -> c.x));
            
            char letter = 'A';
            
            for (Circle circle : uniqueCircles) {
                
                Point textPosition = new Point(circle.x - 10, circle.y + 10);
                Imgproc.putText(original,String.valueOf(circleNumber),textPosition,Imgproc.FONT_HERSHEY_SIMPLEX,0.5,new Scalar(0, 255, 0),1);
                                
                int squareSide = circle.radius * 2;
                int squareX = circle.x - circle.radius;
                int squareY = circle.y - circle.radius;

                Rect squareCircle = new Rect(squareX, squareY, squareSide, squareSide);
                detectedRect.add(squareCircle);

                Mat squareROI = cropped.submat(squareCircle);
                
                double fillPercentage = calculateFillPercentage(squareROI);
                
                if (fillPercentage >= 65.0) {
                    if (answers.length >= 1) {
                        int previousAnswerValue = (int) answers[answers.length - 1][0];

                        if ((circleNumber / 4) == (previousAnswerValue / 4)) {
                            answers[answers.length - 1][1] = "Respuesta no valida";
                        } else {
                            if ((circleNumber / 4) - (previousAnswerValue / 4) == 2) {
                                answers = addAnswer(answers, new Object[]{previousAnswerValue + 4, "Pregunta no respondida"});
                            }
                            answers = addAnswer(answers, new Object[]{circleNumber, letter});
                        }
                    } else {
                        answers = addAnswer(answers, new Object[]{circleNumber, letter});
                    }
                }
                
                circleNumber++;
                letter++;
                
                Imgproc.rectangle(original, squareCircle.tl(), squareCircle.br(), new Scalar(255, 255, 0), 2);

                Imgproc.circle(original, new Point(circle.x, circle.y), circle.radius, new Scalar(0, 0, 255), 2);
            }
        }
        for (Object[] answer : answers) {
            System.out.println(globalCounter + ".- " + "Respuesta detectada  " + answer[0] + "      ->       " + answer[1]);
            System.out.println();
            globalCounter++;
        }
    }
    
    private void idDetection(Mat cropped, Mat original) {
    
    }
    
    public static Object[][] addAnswer(Object[][] original, Object[] newAnswer) {
        // Crear un nuevo arreglo con espacio para el nuevo elemento
        Object[][] newArray = Arrays.copyOf(original, original.length + 1);
        // Agregar el nuevo elemento al final
        newArray[original.length] = newAnswer;
        return newArray;
    }
    
    private double calculateFillPercentage(Mat roi) {
        Mat gray = new Mat();
        if (roi.channels() > 1) {
            Imgproc.cvtColor(roi, gray, Imgproc.COLOR_BGR2GRAY);
        } else {
            gray = roi;
        }

        Mat binary = new Mat();
        Imgproc.threshold(gray, binary, 127, 255, Imgproc.THRESH_BINARY);

        int nonZeroPixels = Core.countNonZero(binary);
        double totalPixels = roi.rows() * roi.cols();

        return (nonZeroPixels / totalPixels) * 100.0;
    }
    
    
    private List<Circle> filterOverlappingCircles(List<Circle> circles) {
        List<Circle> filtered = new ArrayList<>();
        for (Circle current : circles) {
            boolean isOverlapping = false;
            for (Circle other : filtered) {
                double distance = Math.sqrt(Math.pow(current.x - other.x, 2) + Math.pow(current.y - other.y, 2));
                if (distance < (current.radius + other.radius) * 0.5) {
                    isOverlapping = true;
                    break;
                }
            }
            if (!isOverlapping) {
                filtered.add(current);
            }
        }
        return filtered;
    }
    
    static class Circle {
        int x, y, radius;
        boolean isFilled;

        Circle(int x, int y, int radius, boolean isFilled) {
            this.x = x;
            this.y = y;
            this.radius = radius;
            this.isFilled = isFilled;
        }
    }
}