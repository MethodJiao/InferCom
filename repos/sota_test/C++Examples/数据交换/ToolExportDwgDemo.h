/** @class
 *  @brief   文件导入导出工具
 *  @author  北京构力
 *  ------------------------------------------------------------
 *  版本历史       注释                       日期
 *  ------------------------------------------------------------
 *  @version v1.0  初始版本              2021/10/20
 *  ------------------------------------------------------------
 *  @note:  -
 */

class ToolExportDwgDemo
{

public:
	//返回工具ID
	static Utf8CP getToolName();

	//工具启动后响应函数
	static void doExportDwgDemo();
	static void doImportDwgDemo();

private:
	//获取文件名字
	static std::wstring getFileName();

};
