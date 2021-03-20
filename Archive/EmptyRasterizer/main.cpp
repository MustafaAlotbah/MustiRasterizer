

/* Mustafa Graphics Engine  */


/*  
	Setup the window
*/


#include "main.h"


/*
HWND:			handle to a window
				- used to communicate with a particular window (main Window, a dialogue, ...)
HINSTANCE:		handle to a (program) instance
				- a way for the kernel to communicate with the program's instance
HMENU:			handle to a drop down menu.
*/



/*     PRIVATE VARIABLES      */
bool isRunning = true;
mge::VideoBuffer videoBuffer;


/*			FUNCTIONS			*/
LRESULT CALLBACK eventHandler(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);


/*		MACRO FUNCTIONS			*/
// update a button state
#define _update_button(button, vk_btn)\
if (code == vk_btn) {\
input.buttons[button].hasChanged = isDown != input.buttons[button].isDown; \
input.buttons[button].isDown = isDown; \
} 


/*
Goal:	Register window classes to be used by the program
		Create any windows used by the program
		Run the message loop between Windows and the program

hInstance:		current Program
_null:	last instance of the current program (in NT, it's always NULl)
lpszArgument:	contains argc and argv
iCmdShow:		determines whether or not a graphical window should be displayed immediately or if it should run minimized
*/
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE _null, LPSTR lpszArgument, int iCmdShow) {


	/*-----------------------------CREATE & REGISTER WINDOW--------------------------------*/
	// Create a Window class
	WNDCLASS windowClass = {};
	windowClass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	windowClass.hCursor = LoadCursor(NULL, IDC_ARROW);
	windowClass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	windowClass.lpszClassName = L"Graphics Engine build 3";
	windowClass.lpfnWndProc = eventHandler;

	// Register the Window Class
	RegisterClass(&windowClass);


	// create the window and get its handle
	HWND hwnd = CreateWindow(
		windowClass.lpszClassName,
		L"Graphics Engine build 3",
		WS_OVERLAPPEDWINDOW | WS_VISIBLE,
		CW_USEDEFAULT, CW_USEDEFAULT,
		600, 600, 0, 0, hInstance, 0
	);


	// retrieve the handle of device context for GDI
	HDC hdc = GetDC(hwnd);


	/*-----------------------------INITIALIZE PARAMETERS--------------------------------*/
	// Keep track of Delta Time
	// Get the Cycles
	float deltaTime = 0.016;
	LARGE_INTEGER startTime;
	LARGE_INTEGER endTime;
	QueryPerformanceCounter(&startTime);
	// Get the Frequency to transform cycles into time
	float performanceFreq;
	{
		LARGE_INTEGER performance;
		QueryPerformanceFrequency(&performance);
		performanceFreq = (float)(performance.QuadPart);
	}


	// Also keep track of Input
	mge::Input input = {};
	mge::Rasterizer rasterizer(videoBuffer);



	/*-----------------------------PROGRAM LOOP--------------------------------*/
	while (isRunning)
	{

		/*------------- HANDLE MESSAGES AND INPUT --------------*/
		MSG message;
		// update buttons' states
		for (int i = 0; i < mge::ButtonName::BUTTON_COUNT; i++)
		{
			input.buttons[i].hasChanged = false;
		}
		//
		
		// check for changes
		while (PeekMessage(&message, hwnd, 0, 0, PM_REMOVE))
		{
			switch (message.message)
			{
			case WM_KEYUP:
			case WM_KEYDOWN: {

				unsigned int code = (unsigned int)message.wParam;		// which button?
				bool isDown = ((message.lParam & (1 << 31)) == 0);		// down or up?

				_update_button(mge::ButtonName::BUTTON_LEFT, VK_LEFT);
				_update_button(mge::ButtonName::BUTTON_UP, VK_UP);
				_update_button(mge::ButtonName::BUTTON_RIGHT, VK_RIGHT);
				_update_button(mge::ButtonName::BUTTON_DOWN, VK_DOWN);

			} break;

			default:
			{
				// otherwise let windows handle and dispatch it
				TranslateMessage(&message);
				DispatchMessage(&message);
			}
				break;
			}
		}

		/*------------- SIMULATE-------------*/

		rasterizer.OnUpdate(deltaTime);




		/*-------------SHOW BUFFER-------------*/
		StretchDIBits(
			hdc,
			0, 0,
			videoBuffer.width, videoBuffer.height,
			0, 0,
			videoBuffer.width, videoBuffer.height,
			videoBuffer.addr, &(videoBuffer.bitmapInfo),
			DIB_RGB_COLORS, SRCCOPY
		);



		/*------------- UPDATE PARAMETERS --------------*/
		QueryPerformanceCounter(&endTime);
		deltaTime = (endTime.QuadPart - startTime.QuadPart) / performanceFreq;
		startTime = endTime;


	}



	/*-----------------------------SHOULDN'T ARRIVE HERE--------------------------------*/
	return 0;
}




// callback function to handle user-interface and windows messages
LRESULT CALLBACK eventHandler(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
	LRESULT result = 0;

	switch (uMsg)
	{

		// terminating the program
	case WM_CLOSE:
	case WM_DESTROY:
	{
		isRunning = false;
	}
	break;
	// new size to the window
	case WM_SIZE:
	{
		// calculate the new size of the window
		RECT rect;
		GetClientRect(hwnd, &rect);
		videoBuffer.width = rect.right - rect.left;
		videoBuffer.height = rect.bottom - rect.left;

		int buffer_size = videoBuffer.width * videoBuffer.height * sizeof(unsigned int);

		// allocate memory for the buffer
		if (videoBuffer.addr) VirtualFree(videoBuffer.addr, 0, MEM_RELEASE);
		videoBuffer.addr = VirtualAlloc(0, buffer_size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);

		// setup new bitmap info header
		videoBuffer.bitmapInfo.bmiHeader.biSize = sizeof(videoBuffer.bitmapInfo.bmiHeader);
		videoBuffer.bitmapInfo.bmiHeader.biWidth = videoBuffer.width;
		videoBuffer.bitmapInfo.bmiHeader.biHeight = videoBuffer.height;
		videoBuffer.bitmapInfo.bmiHeader.biPlanes = 1;
		videoBuffer.bitmapInfo.bmiHeader.biBitCount = 32;
		videoBuffer.bitmapInfo.bmiHeader.biCompression = BI_RGB;
	}
	break;
	// otherwise
	default:
	{
		result = DefWindowProc(hwnd, uMsg, wParam, lParam);
	}
	break;
	}

	return result;
}