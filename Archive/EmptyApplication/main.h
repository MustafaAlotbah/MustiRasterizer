#pragma once
#ifndef __MAIN_H
#define __MAIN_H





#include<Windows.h>

// Mustafa Graphics Engine
namespace mge
{
	/*		DATATYPES		*/
	typedef struct VideoBufferStruct
	{
		void* addr;			// address of the first pixel (bottom right)
		int width;
		int height;
		BITMAPINFO bitmapInfo;
	} VideoBuffer;

	typedef struct ButtonStateStruct
	{
		bool isDown;
		bool hasChanged;
	} ButtonState;

	enum ButtonName
	{
		BUTTON_UP,
		BUTTON_DOWN,
		BUTTON_LEFT,
		BUTTON_RIGHT,

		BUTTON_COUNT,
	};


	// array of buttons states
	typedef struct InputStruct
	{
		ButtonState buttons[BUTTON_COUNT];
	} Input;




	/* The Application that uses GME API*/


	class Application
	{
	public:
		Application(VideoBuffer* buffer);
		~Application();
		virtual bool OnLoad();
		virtual bool OnUpdate(float DeltaTime);
	private:
		VideoBuffer* videoBuffer;
		LPCWSTR windowTitle;
	};





}





#endif // !__MAIN_H
