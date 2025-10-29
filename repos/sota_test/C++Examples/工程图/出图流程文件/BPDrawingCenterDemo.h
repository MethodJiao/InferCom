#pragma once
/** @class
 *  @brief   控制出图流程
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
	
	class BPDrawingCenterDemo : public IBPDrawingCenterDemo
	{
	public:
		BPDrawingCenterDemo();
		~BPDrawingCenterDemo();
	
		virtual void preProcessing() override  { return; };
		virtual void doDrawingInfo() override;
		virtual void doDrawingCut()override;
		virtual void doDrawingDimension() override;
		virtual void doDrawingTable() override;
		virtual void doDrawingLayout() override;
		virtual void postProcessing()  override;
		
	};
}
