
#pragma once
/** @class
 *  @brief   控制出图流程的模板类
 *  @author  北京构力
 *  ------------------------------------------------------------
 *  版本历史       注释                       日期
 *  ------------------------------------------------------------
 *  @version v1.0  初始版本              2022/4/26
 *  ------------------------------------------------------------
 *  @note:  -
 */


namespace DemoObject
{
	
	class IBPDrawingCenterDemo 
	{
	public:
		IBPDrawingCenterDemo();
		virtual~IBPDrawingCenterDemo();
		void doDrawing();

		virtual void preProcessing() = 0;
		virtual void doDrawingInfo() = 0;;
		virtual void doDrawingCut()=0;
		virtual void doDrawingDimension()=0;
		virtual void doDrawingTable()=0;

		virtual void doDrawingLayout()=0;
		virtual void postProcessing() = 0;

	
	};
	
}
