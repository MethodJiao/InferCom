#include "stdafx.h"
#include "BPCutModelManagerDemo.h"
#include "IBPDrawingCenterDemo.h"
#include "BPDrawingParasManagerDemo.h"


using namespace DemoObject;
IBPDrawingCenterDemo::IBPDrawingCenterDemo()
{

}

IBPDrawingCenterDemo::~IBPDrawingCenterDemo()
{

}

void IBPDrawingCenterDemo::doDrawing()
{
	
	doDrawingCut();
	doDrawingDimension();
}