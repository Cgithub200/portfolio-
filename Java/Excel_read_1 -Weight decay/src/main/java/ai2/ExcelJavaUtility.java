package ai2;

import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.xssf.streaming.SXSSFWorkbook;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.io.IOException;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.*;


public class ExcelJavaUtility {

    private XSSFWorkbook workbook;
    private XSSFSheet sheet;

  
    private int getMaxValue(Map<String, Integer> map) {
        return map.values().stream()
                  .mapToInt(Integer::intValue) // Convert Stream<Integer> to IntStream
                  .max() // Find the maximum value
                  .orElse(0); 
    }
    public static void printMap(Map<String, Integer> map) {
        // Iterate through the entries of the HashMap
        for (Map.Entry<String, Integer> entry : map.entrySet()) {
            String key = entry.getKey();
            Integer value = entry.getValue();
            System.out.println("Key: " + key + ", Value: " + value);
        }
    }
    
    
    public void updateExcelFile(String excelLocation, String outputLocation) {

        int columnCount = getColumnCount();
        int rowCount = getRowCount();
        int[] columnCounter = new int[columnCount];
        
        
        try (FileInputStream fileInputStream = new FileInputStream(excelLocation);
             XSSFWorkbook workbook = new XSSFWorkbook(fileInputStream)) {

            Sheet sheet = workbook.getSheet("Training");
            if (sheet == null) {
                System.out.println("Sheet 'Training' not found in the Excel file.");
                return;
            }

            List<Map<String, Double>> columnMappings = new ArrayList<>();
            
            for (int i = 0; i < columnCount; i++) {
                columnMappings.add(new HashMap<>());
                columnCounter[i] = 0;
            }

            try (SXSSFWorkbook streamingWorkbook = new SXSSFWorkbook()) {
                Sheet streamingSheet = streamingWorkbook.createSheet("Training");

                int batchSize = 10000;
                for (int startRow = 1; startRow < rowCount; startRow += batchSize) {
                	System.out.println("Row: " + startRow);
                    int endRow = Math.min(startRow + batchSize, rowCount);
                    processRows(sheet, streamingSheet, startRow, endRow, columnMappings,columnCounter,false);
          
                }

                try (FileOutputStream fileOutputStream = new FileOutputStream(outputLocation)) {
                    streamingWorkbook.write(fileOutputStream);
                } catch(Exception e) {
                	System.err.println(e);
                }

            } catch (IOException e) {
                e.printStackTrace();
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
        
        outputLocation = "./Data/train3.xlsx";
        try (FileInputStream fileInputStream = new FileInputStream(outputLocation);
                XSSFWorkbook workbook = new XSSFWorkbook(fileInputStream)) {
        	
        	Sheet sheet = workbook.getSheet("Training");
        	if (sheet == null) {
                System.out.println("Sheet 'Training' not found in the Excel file.");
                return;
            }
        	
        	try (SXSSFWorkbook streamingWorkbook = new SXSSFWorkbook()) {
        		Sheet streamingSheet = streamingWorkbook.createSheet("Training");

                int batchSize = 10000;
                for (int startRow = 1; startRow < rowCount; startRow += batchSize) {
                	int endRow = Math.min(startRow + batchSize, rowCount);
                    processRows(sheet, streamingSheet, startRow, endRow, null,columnCounter,true);
                }
                try (FileOutputStream fileOutputStream = new FileOutputStream(outputLocation)) {
                    streamingWorkbook.write(fileOutputStream);
                } catch(Exception e) {
                	System.err.println(e);
                }

                
        	}

        } catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
        
        
        
        System.out.println("done");
    }
    
    private void deleteRow(Sheet sheet, int rowIndex) {
    	Row deletedRow = sheet.getRow(rowIndex);
    	if (deletedRow != null) {
    		sheet.removeRow(deletedRow);
    		int lastRowPos = sheet.getLastRowNum();
    		if (rowIndex < lastRowPos) {
                sheet.shiftRows(rowIndex + 1, lastRowPos, -1);
            }
    	}
    }
    
    private double scaleValue (double value, double inputMin, double inputMax) {
    	double normalizedValue = (value - inputMin) / (inputMax - inputMin);
    	return 0.1 * (normalizedValue * (0.9 - 0.1));
    }

    private void processRows(Sheet inputSheet, Sheet outputSheet, int startRow, int endRow, List<Map<String, Double>> columnMappings, int[] columnCounter, Boolean Normalize) {
        int outputRowIndex = startRow;
        for (int rowPos = startRow; rowPos < endRow; rowPos++) {
            Row row = inputSheet.getRow(rowPos);

            if (row == null) continue;

            Row newRow = outputSheet.createRow(outputRowIndex++);

            for (int columnPos = 0; columnPos < columnCounter.length; columnPos++) {

                Cell cell = row.getCell(columnPos);
                
                
                
                
                if (cell == null) {
                	deleteRow(outputSheet, rowPos);
                	outputRowIndex--;
                	break;
                };

                String cellValue = getCellValue(cell);
              
                if (cellValue != null && cellValue != "") {

                	double mappedValue;
                	Cell newCell = newRow.createCell(columnPos);
                	

	                	try {
	                		mappedValue = Double.valueOf(cellValue);
	                		columnCounter[columnPos]  = (int) Math.ceil( Math.max(mappedValue, columnCounter[columnPos]));
	                		if (Normalize) {  
	                			mappedValue = scaleValue(mappedValue,0,columnCounter[columnPos]);
	                		}
	                		
	                		
	                	} catch(NumberFormatException e) {
	                		mappedValue = columnMappings.get(columnPos).getOrDefault(cellValue, (double) -1);
	
	                        if (mappedValue == -1) {
	                        	mappedValue = columnCounter[columnPos];
	                        	columnCounter[columnPos] += 1;
	                            columnMappings.get(columnPos).put(cellValue, mappedValue);
	                        }
	                	}
                	
                	
                    newCell.setCellValue(mappedValue);
                } else {
                	deleteRow(outputSheet, rowPos);
                	outputRowIndex--;
                	break;
                }
                
            }

        }
    }

    private String getCellValue(Cell cell) {
        switch (cell.getCellType()) {
            case 1:
                return cell.getStringCellValue().trim();
            case 0:
            	double response = cell.getNumericCellValue();
            	if (response < 0) return null;
                return Double.toString(response);
            default:
                return null;
        }
    }

   
    
    
    
    
    
    
    public ExcelJavaUtility(String excelLocation, String sheetName) {
        try {
            workbook = new XSSFWorkbook(excelLocation);
            sheet = workbook.getSheet(sheetName);
        } catch (IOException e) {
            System.err.println("Error opening or reading the Excel file: " + e.getMessage());
        }
    }

    public void changeSheet(String sheetName) {
        if (workbook == null) {
            System.err.println("Workbook is not initialized.");
            return;
        }
        sheet = workbook.getSheet(sheetName);
        if (sheet == null) {
            System.err.println("Sheet not found: " + sheetName);
        }
    }

    public ArrayList<Double> getRowCellData(int rowIndex) {
        ArrayList<Double> rowData = new ArrayList<>();
        if (sheet == null) {
            System.err.println("Sheet is not initialized.");
            return rowData;
        }

        int columnCount = getColumnCount();
        for (int i = 0; i < columnCount; i++) {
            try {
                rowData.add(sheet.getRow(rowIndex + 1).getCell(i).getNumericCellValue());
            } catch (Exception e) {
                System.err.println("Reading cell data error: " + e.getMessage());
            }
        }
        return rowData;
    }

    public int getRowCount() {
        if (sheet == null) {
            System.err.println("Sheet is not initialized.");
            return 0;
        }
        return sheet.getPhysicalNumberOfRows();
    }

    public int getColumnCount() {
        if (sheet == null || sheet.getRow(0) == null) {
            System.err.println("Sheet or column header row is not initialized.");
            return 0;
        }
        return sheet.getRow(0).getPhysicalNumberOfCells();
    }
    
    // Close the workbook to free resources
    public void close() {
        if (workbook != null) {
            try {
                workbook.close();
            } catch (Exception e) {
                System.err.println("Closing the workbook error: " + e.getMessage());
            }
        }
    }
}
