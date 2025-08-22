/*
 * app_error.h
 *
 *  Created on: Aug 21, 2025
 *      Author: rishi_latchmepersad
 */

#ifndef INC_APP_ERROR_H_
#define INC_APP_ERROR_H_

#ifndef ERROR_DEFAULT_MSG
#define ERROR_DEFAULT_MSG  "Fatal error. Rebooting in 5 seconds..."
#endif

void error_handler_with_message(const char *msg);

//default HAL function symbol
void Error_Handler(void);

#endif /* INC_APP_ERROR_H_ */
