#pragma once
/** @class
*  @brief   创建文字
*  @author  北京构力
*  @date    2021/9/22
*  ------------------------------------------------------------
*  版本历史       注释                       日期
*  ------------------------------------------------------------
*  @version v1.0  初始版本              2021/9/22
*  ------------------------------------------------------------
*  @note:  -
*/

class ToolCreateTextTest
{
public:
	ToolCreateTextTest();
	~ToolCreateTextTest();

	static PBBuildingElementProxyPtr createText(PString str, GePoint3d ptOri);

	static BPTextEntityPtr createText2(PString str, GePoint3d ptOri);
};

