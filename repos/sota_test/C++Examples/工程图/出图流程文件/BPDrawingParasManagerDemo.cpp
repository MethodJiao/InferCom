#pragma once
#include "stdafx.h"
#include "BPDrawingParasManagerDemo.h"
/** @class
 *  @brief   出图参数处理
 *  @author  北京构力
 *  ------------------------------------------------------------
 *  版本历史       注释                       日期
 *  ------------------------------------------------------------
 *  @version v1.0  初始版本              2022/4/26
 *  ------------------------------------------------------------
 *  @note:  -
 */
using namespace DemoObject;

BPDrawingParasManagerDemo::BPDrawingParasManagerDemo()
{

}
BPDrawingParasManagerDemo::~BPDrawingParasManagerDemo()
{

}
BPDrawingParasManagerDemoR BPDrawingParasManagerDemo::Get()
{
	static BPDrawingParasManagerDemo single;
	return single;
}

