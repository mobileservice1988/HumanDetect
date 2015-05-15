package liang.example.humandetect;

import java.util.ArrayList;
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.View;
import android.view.View.OnTouchListener;
import android.widget.ImageView;
import android.widget.RelativeLayout;

public class MainActivity extends Activity implements OnTouchListener{

	private final String TAG = "HumanDetect";
		 	 	
	protected static final int   CAMERA_REQUEST = 1;
	protected static final int   GALLERY_REQUEST = 2;
	
	private int 				 width;
	private int 				 height;
	
	private boolean              touchFlag;
	
	private ImageView 			 imageView;
	private ImageView			 tapImageView;	
	private ImageView			 rectImageView;
	
	private RelativeLayout.LayoutParams params;
	private RelativeLayout.LayoutParams rectparams;
	
	private MenuItem             mItemTesting;
	private MenuItem             mItemPeopleRegion;
	private MenuItem             mItemSubProcessing;
	private MenuItem             mItemDetectRegion;
	private MenuItem             mItemContourImage;
    private MenuItem             mItemPosterizeImage;
    private MenuItem             mItemContrastImage;
    private MenuItem             mItemTestImage1;    
    private MenuItem             mItemTestImage2; 
    private MenuItem             mItemTestImage3;
    
    private Bitmap				 testImage;
    private Bitmap               originImage;
    
    private ColorBlobDetector    mDetector;
    private Mat                  mSpectrum;
    private Size                 SPECTRUM_SIZE;
    
    private double[]             BLACK_COLOR;
    private Mat					 tempMat;
    private Mat 				 originalMat;
    private Mat					 testMat;
    
    private Point					seedPoint1;
    private Point					seedPoint2;
    private Point					seedPoint3;
    private Point					seedPoint4;
	@Override
	protected void onCreate(Bundle savedInstanceState) 
	{
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		
		imageView 	 = (ImageView)findViewById(R.id.imageView);
		tapImageView = (ImageView)findViewById(R.id.tapImageView);
		rectImageView = (ImageView)findViewById(R.id.RectImageView);
		
		params = (RelativeLayout.LayoutParams)tapImageView.getLayoutParams();
		params.leftMargin = 500;
		params.topMargin = 50;
		params.width  = 50;
		params.height = 50;
		tapImageView.setLayoutParams(params);
		
		rectparams = (RelativeLayout.LayoutParams)rectImageView.getLayoutParams();
		rectparams.leftMargin = 0;
		rectparams.topMargin = 0;
		rectparams.width  = 0;
		rectparams.height = 0;
		rectImageView.setLayoutParams(rectparams);
		
		
		imageView.setOnTouchListener(this);			
		originImage = BitmapFactory.decodeResource(getResources(), R.drawable.testface1);
		imageView.setImageBitmap(originImage);
		
		if (!OpenCVLoader.initDebug()) {
	        Log.e("HomeActivity", "failed loading OpenCV");
	    } else {
	    	
	    }		
	}

	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		// Inflate the menu; this adds items to the action bar if it is present.
		Log.i(TAG, "called onCreateOptionsMenu");
		mItemTesting   = menu.add("Testing"); 
		mItemPeopleRegion   = menu.add("Region");
		mItemDetectRegion   = menu.add("Detect"); 
		mItemSubProcessing  = menu.add("SubProcessing");
		mItemContourImage   = menu.add("Contour"); 
        mItemContrastImage  = menu.add("Contrast");		     
        mItemPosterizeImage = menu.add("Posterize");
        mItemTestImage1     = menu.add("TestImage1");        
        mItemTestImage2     = menu.add("TestImage2"); 
        mItemTestImage3     = menu.add("TestImage3"); 
		return true;
	}
	
	@Override
	public void onWindowFocusChanged(boolean hasFocus)
	{		
		width  = imageView.getWidth();
		height = imageView.getHeight();
		
	}
	
	@Override
	public boolean onOptionsItemSelected(MenuItem item) {
		
		if (testImage == null) {
			return false;
		}
		Mat imageData = new Mat();     	   	
     	Utils.bitmapToMat(testImage, imageData, true);
     	if (mItemPeopleRegion == item)
     	{
     		Core.rectangle(imageData, new Point( 0, 0), new Point(seedPoint3.x * imageData.cols() / width - 1, seedPoint3.y * imageData.rows()/height), new Scalar(0, 0, 0, 0), -1);
     		Core.rectangle(imageData,  new Point(seedPoint2.x * imageData.cols() / width + 1 , seedPoint2.y* imageData.rows()/height ), new Point(imageData.cols(), imageData.rows()), new Scalar(0, 0, 0, 0), -1);
     		Core.rectangle(imageData, new Point( 0, 0), new Point(imageData.cols(), seedPoint1.y * imageData.rows() / height - 1), new Scalar(0, 0, 0, 0), -1);
     		Core.rectangle(imageData,  new Point(0, seedPoint3.y* imageData.rows() / height + 1), new Point(imageData.cols(), imageData.rows()), new Scalar(0, 0, 0, 0), -1);
     		
     		Utils.matToBitmap(imageData, testImage);
	        
	        
	        Mat scaleMat = new Mat(imageData.rows(), imageData.cols(),imageData.type());
     		Imgproc.resize(imageData, scaleMat, new Size(imageData.cols(), imageData.rows()));
     		Utils.matToBitmap(scaleMat, originImage);
     		imageView.setImageBitmap(originImage);
     		
     		rectparams = (RelativeLayout.LayoutParams)rectImageView.getLayoutParams();
    		rectparams.leftMargin = 0;
    		rectparams.topMargin = 0;
    		rectparams.width  = 0;
    		rectparams.height = 0;
    		rectImageView.setLayoutParams(rectparams);
            
     	} else if (mItemTesting == item) {
     		
     		Mat  resultImage  = new Mat();
         	resultImage = imageData.clone();
         	
     		Mat mIntermediateMat = new Mat();  
     		ArrayList<Mat> channels = new ArrayList<Mat>();
     		Mat blackData = new Mat(imageData.size(), CvType.CV_8UC1, new Scalar( 0));
	     	
     		imageData.convertTo(imageData, imageData.type(), 1.1);   
     		Core.split(imageData, channels);     		
     		for (int i = 0; i < 3; i++) {
     			Imgproc.Canny(channels.get(i), mIntermediateMat, 20, 70);
     			blackData.setTo(new Scalar(255), mIntermediateMat);
     		}    
     		
     		Mat  hierarchy = new Mat(); 
	     	List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
	     	Imgproc.Canny(blackData, mIntermediateMat, 50, 90);
	    	Imgproc.findContours(mIntermediateMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_TC89_L1);
	    	Imgproc.drawContours(blackData, contours, -1, new Scalar(255) , 2);
	    	Imgproc.threshold(blackData, blackData, 20, 255, Imgproc.THRESH_BINARY);
	    	
     		double[] blackColor = new double[]{125};
     		double[] color = new double[1];
     		ArrayList<Point> list = new ArrayList<Point>();
     		ArrayList<Point> list1 = new ArrayList<Point>();
     		
     		list.add( new Point((int)seedPoint1.x * imageData.cols()/width + 3,(int)seedPoint1.y * imageData.rows()/height + 3));
     		blackData.put((int)seedPoint1.x * imageData.cols()/width + 3,(int)seedPoint1.y * imageData.rows()/height + 3, blackColor);
     		list.add( new Point((int)seedPoint2.x * imageData.cols()/width - 3,(int)seedPoint2.y * imageData.rows()/height + 3));
     		blackData.put((int)seedPoint2.x * imageData.cols()/width - 3,(int)seedPoint2.y * imageData.rows()/height + 3, blackColor);
     		
     		while (list.size() > 0) {
     			list1.clear();
     			for (int i = 0; i < list.size(); i++) {
					Point pt = list.get(i);
									
					if (pt.y + 1 < imageData.rows()) {
						color = blackData.get( (int)pt.y + 1 ,(int)pt.x);
						if (color[0] == 0) {
							list1.add(new Point(pt.x , pt.y + 1));
							blackData.put((int)pt.y + 1, (int)pt.x, blackColor);
						} else { 
							blackData.put((int)pt.y + 1, (int)pt.x, blackColor);
						}
					}
					
					if (pt.y - 1 > 0) {
						color = blackData.get( (int)pt.y - 1 ,(int)pt.x);
						if (color[0] == 0) {
							list1.add(new Point(pt.x , pt.y - 1));
							blackData.put((int)pt.y - 1, (int)pt.x, blackColor);
						} else { 
							blackData.put((int)pt.y - 1, (int)pt.x, blackColor);
						}
					}
					
					if (pt.x + 1 < blackData.cols()) {
						color = blackData.get( (int)pt.y, (int)pt.x + 1);
						if (color[0] == 0) {
							list1.add(new Point(pt.x + 1, pt.y));
							blackData.put((int)pt.y, (int)pt.x + 1, blackColor);
						} else { 
							blackData.put((int)pt.y, (int)pt.x + 1, blackColor);
						}							
					}
					if (pt.x - 1 > 0) {
						color = blackData.get( (int)pt.y , (int)pt.x - 1);
						if (color[0] == 0) {
							list1.add(new Point(pt.x - 1, pt.y));
							blackData.put((int)pt.y, (int)pt.x - 1, blackColor);
						} else { 
							blackData.put((int)pt.y, (int)pt.x - 1, blackColor);
						}							
					}
				}
     			list.clear();
     			list.addAll(list1);
			}   
     		
     		Core.rectangle(blackData,  new Point( 0, 0), new Point(seedPoint3.x * imageData.cols() / width + 3, seedPoint3.y * imageData.rows()/height), new Scalar(125), -1);
     		Core.rectangle(blackData,  new Point(seedPoint2.x * imageData.cols() / width - 3 , seedPoint2.y* imageData.rows()/height ), new Point(imageData.cols(), imageData.rows()), new Scalar(125), -1);
     		Core.rectangle(blackData,  new Point( 0, 0), new Point(imageData.cols(), seedPoint1.y * imageData.rows() / height + 3), new Scalar(125), -1);
     		Core.rectangle(blackData,  new Point(0, seedPoint3.y* imageData.rows() / height - 3), new Point(imageData.cols(), imageData.rows()), new Scalar(125), -1);
     		
     		tempMat = blackData.clone();
     		
     		Mat scaleMat = new Mat(originalMat.rows(), originalMat.cols(),blackData.type());
     		Imgproc.resize(blackData, scaleMat, new Size(originalMat.cols(), originalMat.rows()));
     		     		
     		for (int i = 0; i < scaleMat.rows(); i++) {
				for (int j = 0; j < scaleMat.cols(); j++) {
					if (compareGreyPixels(scaleMat.get(i, j), blackColor)) {
						originalMat.put(i, j, BLACK_COLOR);
					}
				}
			}     	
     		
	    	Utils.matToBitmap(originalMat, originImage);
	        imageView.setImageBitmap(originImage);
	        
	        Imgproc.resize(originalMat, testMat, new Size(originalMat.cols() , originalMat.rows() ));                    
            testImage = Bitmap.createBitmap(testMat.cols(), testMat.rows(), originImage.getConfig());
            Utils.matToBitmap(testMat, testImage);
            
		} else if (mItemSubProcessing == item) {
			
			Mat  resultImage  = new Mat();
         	resultImage = imageData.clone();
         	
			double[] blackColor = new double[]{125};
     		double[] color = new double[1];
     		
			ArrayList<Point> list = new ArrayList<Point>();
     		ArrayList<Point> list1 = new ArrayList<Point>();
     		Mat blackData =	tempMat.clone();
     		params = (RelativeLayout.LayoutParams)tapImageView.getLayoutParams();     		
     		int x = (imageData.cols() * (params.leftMargin + params.width)) / width;
	    	int y = (imageData.rows() * (params.topMargin + params.height))  / height ;
	    	list.add(new Point(x, y)); 
	    	blackData.put(y , x, blackColor);
     		while (list.size() > 0) {
     			list1.clear();
     			for (int i = 0; i < list.size(); i++) {     				
					Point pt = list.get(i);
									
					if (pt.y + 1 < imageData.rows()) {
						color = blackData.get( (int)pt.y + 1 ,(int)pt.x);
						if (color[0] == 0) {
							list1.add(new Point(pt.x , pt.y + 1));
							blackData.put((int)pt.y + 1, (int)pt.x, blackColor);
						}
					}
					
					if (pt.y - 1 > 0) {
						color = blackData.get( (int)pt.y - 1 ,(int)pt.x);
						if (color[0] == 0) {
							list1.add(new Point(pt.x , pt.y - 1));
							blackData.put((int)pt.y - 1, (int)pt.x, blackColor);
						}
					}
					
					if (pt.x + 1 < blackData.cols()) {
						color = blackData.get( (int)pt.y, (int)pt.x + 1);
						if (color[0] == 0) {
							list1.add(new Point(pt.x + 1, pt.y));
							blackData.put((int)pt.y, (int)pt.x + 1, blackColor);
						}							
					}
					if (pt.x - 1 > 0) {
						color = blackData.get( (int)pt.y , (int)pt.x - 1);
						if (color[0] == 0) {
							list1.add(new Point(pt.x - 1, pt.y));
							blackData.put((int)pt.y, (int)pt.x - 1, blackColor);
						}							
					}
				}
     			list.clear();
     			list.addAll(list1);
			}
     		tempMat = blackData.clone();
     		
     		Mat scaleMat = new Mat(originalMat.rows(), originalMat.cols(),blackData.type());
     		Imgproc.resize(blackData, scaleMat, new Size(originalMat.cols(), originalMat.rows()));
     		
     		
     		for (int i = 0; i < scaleMat.rows(); i++) {
				for (int j = 0; j < scaleMat.cols(); j++) {
					if (compareGreyPixels(scaleMat.get(i, j), blackColor)) {
						originalMat.put(i, j, BLACK_COLOR);
					}
				}
			}     		
	    	Utils.matToBitmap(scaleMat, originImage);
	        imageView.setImageBitmap(originImage);
	        
	        Imgproc.resize(originalMat, testMat, new Size(originalMat.cols() , originalMat.rows() ));                    
            testImage = Bitmap.createBitmap(testMat.cols(), testMat.rows(), originImage.getConfig());
            Utils.matToBitmap(testMat, testImage);
	        
		} else if (item == mItemPosterizeImage){
	    	
	    	Mat greyImageData = new Mat();
	     	Mat mIntermediateMat = new Mat();  
	    	Imgproc.cvtColor(imageData, greyImageData, Imgproc.COLOR_RGBA2GRAY);
	    	Imgproc.Canny(greyImageData, mIntermediateMat, 50, 90);
	        imageData.setTo(new Scalar(0, 255, 0, 255), mIntermediateMat);
	        Core.convertScaleAbs(imageData, mIntermediateMat, 1./16, 0);
	        Core.convertScaleAbs(mIntermediateMat, imageData, 16, 0);
	        Utils.matToBitmap(imageData, testImage);
	        imageView.setImageBitmap(testImage);
	            
	    } else if (item == mItemDetectRegion){
	    	
	    	Log.i("", "width:" + imageData.cols());
	    	Log.i("", "height:" + imageData.rows());
	    	
	    	Rect touchedRect = new Rect();
	    	touchedRect.x = (imageData.cols() * params.leftMargin) / width;
	    	touchedRect.y = (imageData.rows() * params.topMargin)  / height ;
	    	touchedRect.width = (imageData.cols() * 50)/width;
	    	touchedRect.height = (imageData.rows() * 50) / height;
	    	
	    	detectHumanRegion(imageData , touchedRect);	
	    	imageView.setImageBitmap(testImage);
	    } else if (item == mItemContourImage){	 
	    	
	    	Mat  resultImage  = new Mat();
         	resultImage = imageData.clone();
         	
     		Mat mIntermediateMat = new Mat();  
     		ArrayList<Mat> channels = new ArrayList<Mat>();
     		Mat blackData = new Mat(imageData.size(), CvType.CV_8UC1, new Scalar( 0));
	     	
     		imageData.convertTo(imageData, imageData.type(), 1.1);   
     		Core.split(imageData, channels);     		
     		for (int i = 0; i < 3; i++) {
     			Imgproc.Canny(channels.get(i), mIntermediateMat, 20, 70);
     			blackData.setTo(new Scalar(255), mIntermediateMat);
     		}
     		
	    	Utils.matToBitmap(mIntermediateMat, testImage);
	        imageView.setImageBitmap(testImage);
	    } else if (item == mItemContrastImage){
	    	Mat mIntermediateMat = new Mat(); 
	    	ArrayList<Mat> channels = new ArrayList<Mat>();	     	
     		imageData.convertTo(imageData, imageData.type(), 1.1);   
     		Core.split(imageData, channels);     		
     		for (int i = 0; i < 3; i++) {
     			Imgproc.Canny(channels.get(i), mIntermediateMat, 50, 90);
     			imageData.setTo(new Scalar( 0, 255, 0, 0), mIntermediateMat);
     		} 
     		Core.convertScaleAbs(imageData, mIntermediateMat, 1./16, 0);
	        Core.convertScaleAbs(mIntermediateMat, imageData, 16, 0);
	    	imageData.convertTo(imageData, imageData.type(), 2);
	    	Utils.matToBitmap(imageData, testImage);
	        imageView.setImageBitmap(testImage);
	    } else if (item == mItemTestImage1){
  	
	    	originImage = BitmapFactory.decodeResource(getResources(), R.drawable.testface1);
	    	Utils.bitmapToMat(originImage, originalMat);
            Imgproc.resize(originalMat, testMat, new Size(originalMat.cols(), originalMat.rows()));                    
            testImage = Bitmap.createBitmap(testMat.cols(), testMat.rows(), originImage.getConfig());
            Utils.matToBitmap(testMat, testImage);
            imageView.setImageBitmap(originImage);	 	        
	    }  else if (item == mItemTestImage2){
	    	
	    	originImage = BitmapFactory.decodeResource(getResources(), R.drawable.testface2);
	    	Utils.bitmapToMat(originImage, originalMat);
            Imgproc.resize(originalMat, testMat, new Size(originalMat.cols() , originalMat.rows() ));                    
            testImage = Bitmap.createBitmap(testMat.cols(), testMat.rows(), originImage.getConfig());
            Utils.matToBitmap(testMat, testImage);
            imageView.setImageBitmap(originImage);	 
	    }  else if (item == mItemTestImage3){
	    	
	    	originImage = BitmapFactory.decodeResource(getResources(), R.drawable.testface3);
	    	Utils.bitmapToMat(originImage, originalMat);
            Imgproc.resize(originalMat, testMat, new Size(originalMat.cols() , originalMat.rows() ));                    
            testImage = Bitmap.createBitmap(testMat.cols(), testMat.rows(), originImage.getConfig());
            Utils.matToBitmap(testMat, testImage);
	        imageView.setImageBitmap(originImage);	        
	    } 
	    return true;
	}
	
	private void detectHumanRegion( Mat imageData, Rect touchedRect ) {
		
		Mat  hierarchy = new Mat();   	
     	List<MatOfPoint> contours    = new ArrayList<MatOfPoint>();     	
     	Scalar         mBlobColorHsv = new Scalar(255);     	
     	Mat  resultImage  = new Mat();
     	Utils.bitmapToMat(originImage, resultImage);
        
     	imageData.convertTo(imageData, imageData.type(), 1);
    	Mat touchedRegionHsv = new Mat();
    	Mat touchedRegionRgba = imageData.submat(touchedRect);
        Imgproc.cvtColor(touchedRegionRgba, touchedRegionHsv, Imgproc.COLOR_RGB2HSV_FULL);
        mBlobColorHsv = Core.sumElems(touchedRegionHsv);
        
        int pointCount = touchedRect.width*touchedRect.height;        
        for (int i = 0; i < mBlobColorHsv.val.length; i++)
            mBlobColorHsv.val[i] /= pointCount;
        
        mDetector.setHsvColor(mBlobColorHsv);
        Imgproc.resize(mDetector.getSpectrum(), mSpectrum, SPECTRUM_SIZE);
        mDetector.process(imageData);
        contours = mDetector.getContours();
        Log.e(TAG, "Contours count: " + contours.size());
		        
		Mat whiteData = new Mat(imageData.size(), imageData.type(), new Scalar(255, 255, 255, 255));
		Core.fillPoly(whiteData, contours, new Scalar(0, 0, 0, 0));
		
//		for (int i = 0; i < whiteData.rows(); i++) {				
//			for (int j = 0; j < whiteData.cols(); j++) {
//				if (!comparePixels(whiteData.get(i, j), BLACK_COLOR)) {
//					whiteData.put(i, j, BLACK_COLOR);
//				}else{
//					break;
//				}
//			}
//			for (int j = whiteData.cols() - 1; j > 0; j--) {
//				if (!comparePixels(whiteData.get(i, j), BLACK_COLOR)) {
//					whiteData.put(i, j, BLACK_COLOR);
//				}else{
//					break;
//				}
//			}
//		}			
//				
//		Mat greyData  = new  Mat();
//		
//		ArrayList<MatOfPoint> contours1 = new ArrayList<MatOfPoint>();
//		Imgproc.cvtColor(whiteData, greyData, Imgproc.COLOR_RGBA2GRAY);
//		Imgproc.Canny(greyData, greyData, 50, 100);
//    	Imgproc.findContours(greyData, contours1, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
//    	
//    	for (int i = 0; i < contours1.size(); i++) {
//    		MatOfPoint contour = contours1.get(i);
//    		Rect rt = Imgproc.boundingRect(contour);
//    		if (rt.width > whiteData.cols()/5 && rt.height > whiteData.rows()/5) {
//    			contours1.remove(i);
//			}	    		
//		}
//    	Core.fillPoly(whiteData, contours1, new Scalar(0, 0, 0, 0));
//    	
//    	tempMat = whiteData.clone();
//    	for (int i = 0; i < whiteData.rows(); i++) {			
//			for (int j = whiteData.cols() - 1; j > 0; j--) {
//				if (comparePixels(whiteData.get(i, j), BLACK_COLOR)) {
//					resultImage.put(i, j, BLACK_COLOR);
//				}
//			}
//		}
		
    	Utils.matToBitmap(whiteData, testImage, true);
	}
	
	private boolean compareGreyPixels(double[] firstValue, double[] secondValue){
		if (firstValue[0] == secondValue[0]) {
			return true;
		}
		return false;
	}
	
	
	private boolean comparePixels(double[] firstValue, double[] secondValue){
		if (firstValue[0] == secondValue[0] && firstValue[1] == secondValue[1] && firstValue[2] == secondValue[2] &&firstValue[3] == secondValue[3]) {
			return true;
		}
		return false;		
	}
	
	private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
		
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                
                    Log.i(TAG, "OpenCV loaded successfully");      
            		mDetector = new ColorBlobDetector();
            		mSpectrum = new Mat();            		
                    SPECTRUM_SIZE = new Size(200, 64);
                    BLACK_COLOR   = new double[]{0, 0, 0, 0};
                    
                    originalMat = new Mat();
                    testMat = new Mat();                    
                    Utils.bitmapToMat(originImage, originalMat);
                    Imgproc.resize(originalMat, testMat, new Size(originalMat.cols() , originalMat.rows() ));                    
                    testImage = Bitmap.createBitmap(testMat.cols(), testMat.rows(), originImage.getConfig());
                    Utils.matToBitmap(testMat, testImage);
                    break;
                default:                
                    super.onManagerConnected(status);
                    break;
            }
        }
    };
    
	@Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
    }

	@Override
	public boolean onTouch(View v, MotionEvent event) {
		// TODO Auto-generated method stub
		
		int eid = event.getAction();
		
		int x = (int) event.getX();
		int y = (int) event.getY();
		
		switch (eid) {
		case MotionEvent.ACTION_DOWN:
			
			rectparams.leftMargin = x;
			rectparams.topMargin = y;
			
			break;
		case MotionEvent.ACTION_MOVE:
			
			rectparams.width = x - rectparams.leftMargin;
			rectparams.height = y - rectparams.topMargin;
			rectImageView.setLayoutParams(rectparams);
			

			break;
		case MotionEvent.ACTION_UP:
							
			params.leftMargin = x;
			params.topMargin = y;
			tapImageView.setLayoutParams(params);
			
			if (rectparams.width != 0 && rectparams.height != 0) {
				seedPoint1 = new Point(rectparams.leftMargin, rectparams.topMargin);
				seedPoint2 = new Point(rectparams.leftMargin + rectparams.width, rectparams.topMargin);
				seedPoint3 = new Point(rectparams.leftMargin, rectparams.topMargin + rectparams.height);
				seedPoint4 = new Point(rectparams.leftMargin + rectparams.width, rectparams.topMargin + rectparams.height);
			}
			
			break;
		}
		return true;
	}
}
